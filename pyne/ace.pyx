"""This module is for reading ACE-format cross sections. ACE stands for "A
Compact ENDF" format and originated from work on MCNP_. It is used in a number
of other Monte Carlo particle transport codes.

ACE-format cross sections are typically generated from ENDF_ files through a
cross section processing program like NJOY_. The ENDF data consists of tabulated
thermal data, ENDF/B resonance parameters, distribution parameters in the
unresolved resonance region, and tabulated data in the fast region. After the
ENDF data has been reconstructed and Doppler-broadened, the ACER module
generates ACE-format cross sections.

.. _MCNP: https://laws.lanl.gov/vhosts/mcnp.lanl.gov/
.. _NJOY: http://t2.lanl.gov/codes.shtml
.. _ENDF: http://www.nndc.bnl.gov/endf

.. moduleauthor:: Paul Romano <paul.k.romano@gmail.com>, Anthony Scopatz <scopatz@gmail.com>
"""

from __future__ import division, unicode_literals
import struct
from warnings import warn
from collections import OrderedDict

cimport numpy as np
import numpy as np
from bisect import bisect_right

from pyne cimport nucname
from pyne import nucname
from pyne.rxname import label
from pyne.endf import Tab1

# fromstring func should depend on numpy verison
from pyne._utils import fromstring_split, fromstring_token
cdef bint NP_LE_V15 = int(np.__version__.split('.')[1]) <= 5 and np.__version__.startswith('1')


class Library(object):
    """A Library objects represents an ACE-formatted file which may contain
    multiple tables with data.

    Parameters
    ----------
    filename : str
        Path of the ACE library file to load.

    :attributes:
      **binary** : bool
        Identifies Whether the library is in binary format or not

      **tables** : dict
        Dictionary whose keys are the names of the ACE tables and whose values
        are the instances of subclasses of AceTable (e.g. NeutronTable)

      **verbose** : bool
        Determines whether output is printed to the stdout when reading a
        Library

    """

    def __init__(self, filename):
        # Determine whether file is ASCII or binary
        try:
            self.f = open(filename, 'r')
            # Grab 10 lines of the library
            s = ''.join([self.f.readline() for i in range(10)])

            # Try to decode it with ascii
            sd = s.decode('ascii')

            # No exception so proceed with ASCII
            self.f.seek(0)
            self.binary = False
        except UnicodeDecodeError:
            self.f.close()
            self.f = open(filename, 'rb')
            self.binary = True

        # Set verbosity
        self.verbose = False
        self.tables = {}

    def read(self, table_names=None):
        """read(table_names=None)

        Read through and parse the ACE-format library.

        Parameters
        ----------
        table_names : None, str, or iterable, optional
            Tables from the file to read in.  If None, reads in all of the 
            tables. If str, reads in only the single table of a matching name.
        """
        if isinstance(table_names, basestring):
            table_names = [table_names]

        if table_names is not None:
            table_names = set(table_names)

        if self.binary:
            self._read_binary(table_names)
        else:
            self._read_ascii(table_names)

    def _read_binary(self, table_names, recl_length=4096, entries=512):
        while True:
            start_position = self.f.tell()

            # Check for end-of-file
            if self.f.read(1) == '':
                return
            self.f.seek(start_position)

            # Read name, atomic mass ratio, temperature, date, comment, and
            # material
            name, awr, temp, date, comment, mat = \
                struct.unpack(str('=10sdd10s70s10s'), self.f.read(116))
            name = name.strip()

            # Read ZAID/awr combinations
            data = struct.unpack(str('=' + 16*'id'), self.f.read(192))

            # Read NXS
            nxs = list(struct.unpack(str('=16i'), self.f.read(64)))

            # Determine length of XSS and number of records
            length = nxs[0]
            n_records = (length + entries - 1)/entries

            # verify that we are suppossed to read this table in
            if (table_names is not None) and (name not in table_names):
                self.f.seek(start_position + recl_length*(n_records + 1))
                continue

            # ensure we have a valid table type
            if 0 == len(name) or name[-1] not in table_types:
                # TODO: Make this a proper exception.
                print("Unsupported table: " + name)
                self.f.seek(start_position + recl_length*(n_records + 1))
                continue

            # get the table
            table = table_types[name[-1]](name, awr, temp)

            if self.verbose:
                temp_in_K = round(temp * 1e6 / 8.617342e-5)
                print("Loading nuclide {0} at {1} K".format(name, temp_in_K))
            self.tables[name] = table

            # Read JXS
            table.jxs = list(struct.unpack(str('=32i'), self.f.read(128)))

            # Read XSS
            self.f.seek(start_position + recl_length)
            table.xss = list(struct.unpack(str('={0}d').format(length),
                                           self.f.read(length*8)))

            # Insert empty object at beginning of NXS, JXS, and XSS
            # arrays so that the indexing will be the same as
            # Fortran. This makes it easier to follow the ACE format
            # specification.
            table.nxs = nxs
            table.nxs.insert(0, 0)
            table.nxs = np.array(table.nxs, dtype=int)

            table.jxs.insert(0, 0)
            table.jxs = np.array(table.jxs, dtype=int)

            table.xss.insert(0, 0.0)
            table.xss = np.array(table.xss, dtype=float)

            # Read all data blocks
            table._read_all()

            # Advance to next record
            self.f.seek(start_position + recl_length*(n_records + 1))

    def _read_ascii(self, table_names):
        cdef list lines, rawdata

        f = self.f
        tables_seen = set()
    
        lines = [f.readline() for i in range(13)]

        while (0 != len(lines)) and (lines[0] != ''):
            # Read name of table, atomic mass ratio, and temperature. If first
            # line is empty, we are at end of file
            words = lines[0].split()
            name = words[0]
            awr = float(words[1])
            temp = float(words[2])

            datastr = '0 ' + ' '.join(lines[6:8])
            nxs = fromstring_split(datastr, dtype=int)

            n_lines = (nxs[1] + 3)/4
            n_bytes = len(lines[-1]) * (n_lines - 2) + 1

            # Ensure that we have more tables to read in
            if (table_names is not None) and (table_names < tables_seen):
                break
            tables_seen.add(name)

            # verify that we are suppossed to read this table in
            if (table_names is not None) and (name not in table_names):
                f.seek(n_bytes, 1)
                f.readline()
                lines = [f.readline() for i in range(13)]
                continue

            # ensure we have a valid table type
            if 0 == len(name) or name[-1] not in table_types:
                warn("Unsupported table: " + name, RuntimeWarning)
                f.seek(n_bytes, 1)
                f.readline()
                lines = [f.readline() for i in range(13)]
                continue

            # read and and fix over-shoot
            lines += f.readlines(n_bytes)
            if 12+n_lines < len(lines):
                goback = sum([len(line) for line in lines[12+n_lines:]])
                lines = lines[:12+n_lines]
                f.seek(-goback, 1)

            # get the table
            table = table_types[name[-1]](name, awr, temp)

            if self.verbose:
                temp_in_K = round(temp * 1e6 / 8.617342e-5)
                print("Loading nuclide {0} at {1} K".format(name, temp_in_K))
            self.tables[name] = table

            # Read comment
            table.comment = lines[1].strip()

            # Add NXS, JXS, and XSS arrays to table
            # Insert empty object at beginning of NXS, JXS, and XSS
            # arrays so that the indexing will be the same as
            # Fortran. This makes it easier to follow the ACE format
            # specification.
            table.nxs = nxs

            datastr = '0 ' + ' '.join(lines[8:12])
            table.jxs = fromstring_split(datastr, dtype=int)

            datastr = '0.0 ' + ''.join(lines[12:12+n_lines])
            if NP_LE_V15:
                #table.xss = np.fromstring(datastr, sep=" ")
                table.xss = fromstring_split(datastr, dtype=float)
            else:
                table.xss = fromstring_token(datastr, inplace=True, maxsize=4*n_lines+1)

            # Read all data blocks
            table._read_all()
            lines = [f.readline() for i in range(13)]

        f.seek(0)

    def find_table(self, name):
        """find_table(name)

        Returns a cross-section table with a given name.

        Parameters
        ----------
        name : str
            Name of the cross-section table, e.g. 92235.70c

        """
        return self.tables.get(name, None)

    def __del__(self):
        self.f.close()


class AceTable(object):
    """Abstract superclass of all other classes for cross section tables."""

    def __init__(self, name, awr, temp):
        self.name = name
        self.awr = awr
        self.temp = temp

    def _read_all(self):
        raise NotImplementedError

    def _get_energy_distribution(self, location_dist, location_start):
        """Returns an EnergyDistribution object from data read in starting at
        location_start.
        """

        # Set starting index for energy distribution
        idx = location_dist + location_start - 1

        location_next_law = int(self.xss[idx])
        law = int(self.xss[idx + 1])
        location_data = int(self.xss[idx + 2])

        # Probability of law valifity
        p_law_validity = Tab1.from_ndarray(self.xss, idx + 3)

        # Position index for reading law data
        idx = location_dist + location_data - 1

        # Chcek for valid and supported ACE law
        if law not in _distributions:
            raise IOError("Unsupported ACE secondary energy "
                          "distribution law {0}".format(law))

        # Create instantiation of corresponding class
        edist = _distributions[law]()

        # Parse energy distribution data
        if law in (4, 44, 61, 67):
            edist.read(self.xss, idx, location_dist)
        else:
            edist.read(self.xss, idx)

        # Set law and probability of law validity
        edist.law = law
        edist.p_law_validity = p_law_validity

        # Read next law if present
        if location_next_law > 0:
            edist.next = self._get_energy_distribution(
                location_dist, location_next_law)

        return edist


class NeutronTable(AceTable):
    """A NeutronTable object contains continuous-energy neutron interaction data
    read from an ACE-formatted Type I table. These objects are not normally
    instantiated by the user but rather created when reading data using a
    Library object and stored within the ``tables`` attribute of a Library
    object.

    Parameters
    ----------
    name : str
        ZAID identifier of the table, e.g. '92235.70c'.
    awr : float
        Atomic mass ratio of the target nuclide.
    temp : float
        Temperature of the target nuclide in eV.
    
    :Attributes:
      **awr** : float
        Atomic mass ratio of the target nuclide.

      **energy** : list of floats
        The energy values (MeV) at which reaction cross-sections are tabulated.

      **name** : str
        ZAID identifier of the table, e.g. 92235.70c.

      **nu_p_energy** : list of floats
        Energies in MeV at which the number of prompt neutrons emitted per
        fission is tabulated.

      **nu_p_type** : str
        Indicates how number of prompt neutrons emitted per fission is
        stored. Can be either "polynomial" or "tabular".

      **nu_p_value** : list of floats
        The number of prompt neutrons emitted per fission, if data is stored in
        "tabular" form, or the polynomial coefficients for the "polynomial"
        form.

      **nu_t_energy** : list of floats
        Energies in MeV at which the total number of neutrons emitted per
        fission is tabulated.

      **nu_t_type** : str
        Indicates how total number of neutrons emitted per fission is
        stored. Can be either "polynomial" or "tabular".

      **nu_t_value** : list of floats
        The total number of neutrons emitted per fission, if data is stored in
        "tabular" form, or the polynomial coefficients for the "polynomial"
        form.

      **reactions** : list of Reactions
        A list of Reaction instances containing the cross sections, secondary
        angle and energy distributions, and other associated data for each
        reaction for this nuclide.

      **sigma_a** : list of floats
        The microscopic absorption cross section for each value on the energy
        grid.

      **sigma_t** : list of floats
        The microscopic total cross section for each value on the energy grid.

      **temp** : float
        Temperature of the target nuclide in eV.

    """

    def __init__(self, name, awr, temp):
        super(NeutronTable, self).__init__(name, awr, temp)
        self.reactions = OrderedDict()
        self.photon_reactions = OrderedDict()

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Continuous-E Neutron Table: {0}>".format(self.name)
        else:
            return "<ACE Continuous-E Neutron Table>"

    def _read_all(self):
        self._read_cross_sections()
        self._read_nu()
        self._read_angular_distributions()
        self._read_energy_distributions()
        self._read_gpd()
        self._read_photon_cross_sections()
        self._read_photon_angular_distributions()
        self._read_photon_energy_distributions()
        self._read_yp()
        self._read_fis()
        self._read_unr()

    def _read_cross_sections(self):
        """Reads and parses the ESZ, MTR, LQR, TRY, LSIG, and SIG blocks. These
        blocks contain the energy grid, all reaction cross sections, the total
        cross section, average heating numbers, and a list of reactions with
        their Q-values and multiplicites.
        """

        cdef int n_energies, n_reactions, loc

        # Determine number of energies on nuclide grid and number of reactions
        # excluding elastic scattering
        n_energies = self.nxs[3]
        n_reactions = self.nxs[4]

        # Read energy grid and total, absorption, elastic scattering, and
        # heating cross sections -- note that this appear separate from the rest
        # of the reaction cross sections
        arr = self.xss[self.jxs[1]:self.jxs[1] + 5*n_energies]
        arr.shape = (5, n_energies)
        self.energy, self.sigma_t, self.sigma_a, sigma_el, self.heating = arr

        # Create elastic scattering reaction
        elastic_scatter = Reaction(2, self)
        elastic_scatter.Q = 0.0
        elastic_scatter.IE = 1
        elastic_scatter.multiplicity = 1
        elastic_scatter.sigma = sigma_el
        self.reactions[2] = elastic_scatter

        # Create all other reactions with MT values
        mts = np.asarray(self.xss[self.jxs[3]:self.jxs[3] + n_reactions], dtype=int)
        qvalues = np.asarray(self.xss[self.jxs[4]:self.jxs[4] + 
                                      n_reactions], dtype=float)
        tys = np.asarray(self.xss[self.jxs[5]:self.jxs[5] + n_reactions], dtype=int)

        # Create all reactions other than elastic scatter
        reactions = [(mt, Reaction(mt, self)) for mt in mts]
        self.reactions.update(reactions)

        # Loop over all reactions other than elastic scattering
        for i, reaction in enumerate(self.reactions.values()[1:]):
            # Copy Q values and multiplicities and determine if scattering
            # should be treated in the center-of-mass or lab system
            reaction.Q = qvalues[i]
            reaction.multiplicity = abs(tys[i])
            reaction.center_of_mass = (tys[i] < 0)

            # Get locator for cross-section data
            loc = int(self.xss[self.jxs[6] + i])

            # Determine starting index on energy grid
            reaction.IE = int(self.xss[self.jxs[7] + loc - 1])

            # Determine number of energies in reaction
            n_energies = int(self.xss[self.jxs[7] + loc])

            # Read reaction cross section
            reaction.sigma = self.xss[self.jxs[7] + loc + 1:
                                          self.jxs[7] + loc + 1 + n_energies]

    def _read_nu(self):
        """Read the NU block -- this contains information on the prompt
        and delayed neutron precursor yields, decay constants, etc
        """
        cdef int idx, i, jxs2, KNU, LNU, NR, NE, NC

        self.nu = {}

        # No NU block
        jxs2 = self.jxs[2]
        if jxs2 == 0:
            return

        # Either prompt nu or total nu is given
        if self.xss[jxs2] > 0:
            KNU = jxs2
            LNU = int(self.xss[KNU])

            whichnu = 'prompt' if self.jxs[24] > 0 else 'total'
            self.nu[whichnu] = {}

            if LNU == 1:
                # Polynomial function form of nu
                self.nu[whichnu]['form'] = 'polynomial'
                NC = int(self.xss[KNU+1])
                self.nu[whichnu]['coefficients'] = self.xss[KNU+2:KNU+2+NC]
            elif LNU == 2:
                # Tabular data form of nu
                self.nu[whichnu]['form'] = 'tabular'
                self.nu[whichnu]['values'] = Tab1.from_ndarray(self.xss, KNU + 1)

        # Both prompt nu and total nu
        elif self.xss[jxs2] < 0:
            KNU = jxs2 + 1
            LNU = int(self.xss[KNU])
            self.nu['prompt'] = {}

            if LNU == 1:
                # Polynomial function form of nu
                self.nu['prompt']['form'] = 'polynomial'
                NC = int(self.xss[KNU+1])
                self.nu['prompt']['coefficients'] = self.xss[KNU+2:KNU+2+NC]
            elif LNU == 2:
                # Tabular data form of nu
                self.nu['prompt']['form'] = 'tabular'
                self.nu['prompt']['values'] = Tab1.from_ndarray(self.xss, KNU + 1)

            KNU = jxs2 + int(abs(self.xss[jxs2])) + 1
            LNU = int(self.xss[KNU])
            self.nu['total'] = {}

            if LNU == 1:
                # Polynomial function form of nu
                self.nu['total']['form'] = 'polynomial'
                NC = int(self.xss[KNU+1])
                self.nu['total']['coefficients'] = self.xss[KNU+2:KNU+2+NC]
            elif LNU == 2:
                # Tabular data form of nu
                self.nu['total']['form'] = 'tabular'
                self.nu['total']['values'] = Tab1.from_ndarray(self.xss, KNU + 1)

        # Check for delayed nu data
        if self.jxs[24] > 0:
            self.nu['delayed'] = {}
            self.nu['delayed']['form'] = 'tabular'
            KNU = self.jxs[24]
            self.nu['delayed']['values'] = Tab1.from_ndarray(self.xss, KNU + 1)

            # Delayed neutron precursor distribution
            self.nu['delayed']['precursor_const'] = []
            self.nu['delayed']['precursor_prob'] = []
            i = self.jxs[25]
            n_group = self.nxs[8]
            self.nu['delayed']['n_precursor_groups'] = n_group
            for group in range(n_group):
                self.nu['delayed']['precursor_const'].append(self.xss[i])
                self.nu['delayed']['precursor_prob'].append(
                    Tab1.from_ndarray(self.xss, i + 1))

                # Advance position
                nr = int(self.xss[i + 1])
                ne = int(self.xss[i + 2 + 2*nr])
                i += 3 + 2*nr + 2*ne

            # Energy distribution for delayed fission neutrons
            LED = self.jxs[26]
            self.nu['delayed']['energy_dist'] = []
            for group in range(n_group):
                location_start = int(self.xss[LED + group])
                energy_dist = self._get_energy_distribution(
                    self.jxs[27], location_start)
                self.nu['delayed']['energy_dist'].append(energy_dist)

    def _read_angular_distributions(self):
        """Find the angular distribution for each reaction MT
        """
        cdef int idx, i, j, n_reactions, n_energies, n_bins
        cdef dict ang_cos, ang_pdf, ang_cdf

        # Number of reactions with secondary neutrons (including elastic
        # scattering)
        n_reactions = self.nxs[5] + 1

        # Angular distribution for all reactions with secondary neutrons
        for i, reaction in enumerate(self.reactions.values()[:n_reactions]):
            loc = int(self.xss[self.jxs[8] + i])

            # Check if angular distribution data exist 
            if loc == -1:
                # Angular distribution data are specified through LAWi
                # = 44 in the DLW block
                continue
            elif loc == 0:
                # No angular distribution data are given for this
                # reaction, isotropic scattering is asssumed (in CM if
                # TY < 0 and in LAB if TY > 0)
                reaction.angular_dist = AngularDistribution.isotropic(
                    np.array([self.energy[0], self.energy[-1]]))
                continue

            idx = self.jxs[9] + loc - 1

            reaction.angular_dist = AngularDistribution()
            reaction.angular_dist.read(self.xss, idx, self.jxs[9])

    def _read_energy_distributions(self):
        """Determine the energy distribution for secondary neutrons for
        each reaction MT
        """
        cdef int i

        # Number of reactions with secondary neutrons other than elastic
        # scattering. For elastic scattering, the outgoing energy can be
        # determined from kinematics.
        n_reactions = self.nxs[5]

        for i, reaction in enumerate(self.reactions.values()[1:n_reactions + 1]):
            # Determine locator for ith energy distribution
            location_start = int(self.xss[self.jxs[10] + i])

            # Read energy distribution data
            reaction.energy_dist = self._get_energy_distribution(
                self.jxs[11], location_start)

    def _read_gpd(self):
        """Read total photon production cross section.
        """
        cdef int idx, jxs12, NE

        jxs12 = self.jxs[12]
        if jxs12 != 0:
            # Determine number of energies
            NE = self.nxs[3]

            # Read total photon production cross section
            idx = jxs12
            self.sigma_photon = self.xss[idx:idx+NE]

            # The MCNP manual also specifies that this block contains secondary
            # photon energies based on a 30x20 matrix formulation. However, the
            # ENDF/B-VII.0 libraries distributed with MCNP as well as other
            # libraries do not contain this 30x20 matrix.

            # # The following energies are the discrete incident neutron energies
            # # for which the equiprobable secondary photon outgoing energies are
            # # given
            # self.e_in_photon_equi = np.array(
            #                         [1.39e-10, 1.52e-7, 4.14e-7, 1.13e-6, 3.06e-6,
            #                          8.32e-6,  2.26e-5, 6.14e-5, 1.67e-4, 4.54e-4,
            #                          1.235e-3, 3.35e-3, 9.23e-3, 2.48e-2, 6.76e-2,
            #                          0.184,    0.303,   0.500,   0.823,   1.353,
            #                          1.738,    2.232,   2.865,   3.68,    6.07,
            #                          7.79,     10.,     12.,     13.5,    15.])

            # # Read equiprobable outgoing photon energies
            # # Equiprobable outgoing photon energies for incident neutron
            # # energy i
            # e_out_photon_equi = self.xss[idx:idx+600]
            # if len(e_out_photon_equi) == 600:
            #     self.e_out_photon_equi = e_out_photon_equi
            #     self.e_out_photon_equi.shape = (30, 20)

    def _read_photon_cross_sections(self):
        """Read cross sections for each photon-production reaction"""

        n_photon_reactions = self.nxs[6]
        mts = np.asarray(self.xss[self.jxs[13]:self.jxs[13] + 
                                  n_photon_reactions], dtype=int)

        reactions = [(mt, Reaction(mt, self)) for mt in mts]
        self.photon_reactions.update(reactions)

        for i, rxn in enumerate(self.photon_reactions.values()):
            loca = int(self.xss[self.jxs[14] + i])
            idx = self.jxs[15] + loca - 1
            rxn.mftype = int(self.xss[idx])
            idx += 1

            if rxn.mftype == 12 or rxn.mftype == 16:
                # Yield data taken from ENDF File 12 or 6
                rxn.mtmult = int(self.xss[idx])

                # Read photon yield as function of energy
                rxn.photon_yield = Tab1.from_ndarray(self.xss, idx + 1)

            elif rxn.mftype == 13:
                # Cross section data from ENDF File 13

                # Energy grid index at which data starts
                rxn.IE = int(self.xss[idx])

                # Cross sections
                n_energy = int(self.xss[idx + 1])
                rxn.sigma = self.xss[idx + 2:idx + 2 + n_energy]
            else:
                raise ValueError("MFTYPE must be 12, 13, 16. Got {0}".format(
                        rxn.mftype))


    def _read_photon_angular_distributions(self):
        # Number of reactions
        n_reactions = self.nxs[6]

        # Angular distribution for all reactions with secondary photons
        for i, rxn in enumerate(self.photon_reactions.values()):
            loc = int(self.xss[self.jxs[16] + i])

            if loc == 0:
                # No angular distribution data are given for this reaction,
                # isotropic scattering is asssumed in LAB
                if rxn.mftype == 13:
                    energies = np.array([self.energy[0], self.energy[-1]])
                else:
                    energies = np.array([rxn.photon_yield.x[0],
                                         rxn.photon_yield.x[-1]])

                rxn.angular_dist = AngularDistribution.isotropic(energies)
                continue

            idx = self.jxs[17] + loc - 1

            rxn.angular_dist = AngularDistribution()
            rxn.angular_dist.read(self.xss, idx, self.jxs[17])

    def _read_photon_energy_distributions(self):
        """Determine the energy distributions for secondary photons"""

        # Number of reactions with secondary photons
        n_reactions = self.nxs[6]

        for i, rxn in enumerate(self.photon_reactions.values()):
            # Determine locator for ith energy distribution
            location_start = int(self.xss[self.jxs[18] +i])

            # Read energy distribution data
            rxn.energy_dist = self._get_energy_distribution(
                self.jxs[19], location_start)

    def _read_yp(self):
        """Read list of reactions required as photon production yield
        multipliers.
        """
        if self.nxs[6] != 0:
            idx = self.jxs[20]
            NYP = int(self.xss[idx])
            if NYP > 0:
                dat = np.asarray(self.xss[idx+1:idx+1+NYP], dtype=int)
                self.MT_for_photon_yield = dat

    def _read_fis(self):
        """Read total fission cross-section data if present. Generally,
        this table is not provided since it is redundant.
        """
        # Check if fission block is present
        idx = self.jxs[21]
        if idx == 0:
            return

        # Read fission cross sections
        self.IE_fission = int(self.xss[idx])  # Energy grid index
        NE = int(self.xss[idx+1])
        self.sigma_f = self.xss[idx+2:idx+2+NE]

    def _read_unr(self):
        """Read the unresolved resonance range probability tables if present.
        """
        cdef int idx, N, M, INT, ILF, IOA, IFF

        # Check if URR probability tables are present
        idx = self.jxs[23]
        if idx == 0:
            return

        N = int(self.xss[idx])     # Number of incident energies
        M = int(self.xss[idx+1])   # Length of probability table
        INT = int(self.xss[idx+2]) # Interpolation parameter (2=lin-lin, 5=log-log)
        ILF = int(self.xss[idx+3]) # Inelastic competition flag
        IOA = int(self.xss[idx+4]) # Other absorption flag
        IFF = int(self.xss[idx+5]) # Factors flag
        idx += 6

        self.urr_energy = self.xss[idx:idx+N] # Incident energies
        idx += N

        # Set up URR probability table
        urr_table = self.xss[idx:idx+N*6*M]
        urr_table.shape = (N, 6, M)
        self.urr_table = urr_table

    def find_reaction(self, mt):
        return self.reactions.get(mt, None)

    def __iter__(self):
        # Generators not supported in Cython
        #for r in self.reactions.values():
        #    yield r
        return iter(self.reactions.values())


class AngularDistribution(object):
    def __init__(self):
        pass

    @classmethod
    def isotropic(cls, energy):
        pass

    def read(self, array, idx, loc_and):
        # Number of energies at which angular distributions are tabulated
        n_energies = int(array[idx])
        idx += 1

        # Incoming energy grid
        self.energy = array[idx:idx + n_energies]
        idx += n_energies

        # Read locations for angular distributions
        lc = np.asarray(array[idx:idx + n_energies], dtype=int)
        idx += n_energies

        self.interp = np.zeros(n_energies, dtype=int)
        self.cosine = []
        self.pdf = []
        self.cdf = []
        for i in range(n_energies):
            if lc[i] > 0:
                # Equiprobable 32 bin distribution
                idx = loc_and + abs(lc[i]) - 1
                cos = array[idx:idx + 33]
                pdf = np.zeros(33)
                pdf[:32] = 1.0/(32.0*(cos[1:] - cos[:-1]))
                cdf = np.linspace(0.0, 1.0, 33)

                self.interp[i] = 2
                self.cosine.append(cos)
                self.pdf.append(pdf)
                self.cdf.append(cdf)
            elif lc[i] < 0:
                # Tabular angular distribution
                idx = loc_and + abs(lc[i]) - 1
                self.interp[i] = int(array[idx])
                n_points = int(array[idx + 1])
                data = array[idx + 2:idx + 2 + 3*n_points]
                data.shape = (3, n_points)
                self.cosine.append(data[0])
                self.pdf.append(data[1])
                self.cdf.append(data[2])
            else:
                # Isotropic angular distribution
                self.interp[i] = 1
                self.cosine.append(np.array([-1., 0., 1.]))
                self.pdf.append(np.array([0.5, 0.5, 0.5]))
                self.cdf.append(np.array([0., 0.5, 1.]))


class EnergyDistribution(object):
    def __init__(self):
        pass

    def read(self, array, idx):
        raise NotImplementedError


class Law1(EnergyDistribution):
    def __init__(self):
        pass

    def read(self, array, idx):
        n_regions = int(array[idx])
        idx += 1
        if n_regions > 0:
            data = np.asarray(array[idx:idx + 2*n_regions], dtype=int)
            data.shape = (2, n_regions)
            self.nbt, self.interp = data
            idx += 2 * n_regions
      
        # Number of outgoing energies in each E_out table
        NE = int(array[idx])
        self.energy_in = array[idx + 1:idx + 1 + NE]
        idx += 1 + NE

        # Read E_out tables
        NET = int(array[idx])
        idx += 1
        self.energy_out = np.zeros(NE, NET)
        for i in range(NE):
            self.energy_out[NE,:] = array[idx:idx + NET]
            idx += NET


class Law2(EnergyDistribution):
    """Discrete photon energy distribution"""

    def __init__(self):
        pass

    def read(self, array, idx):
        self.lp = int(array[idx])
        self.eg = array[idx + 1]


class Law3(EnergyDistribution):
    """Level inelastic scattering"""

    def __init__(self):
        pass

    def read(self, array, idx):
        self.lab_energy_threshold, self.mass_ratio = array[idx:idx + 2]

class Law4(EnergyDistribution):
    """Continuous tabular distribution (ENDF Law 1)"""

    def __init__(self):
        pass

    def read(self, array, idx, ldis):
        # Read number of interpolation regions and incoming energies
        n_regions = int(array[idx])
        n_energy_in = int(array[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            self.nbt = np.asarray(array[idx:idx + n_regions], dtype=int)
            self.interp = np.asarray(array[idx + n_regions:
                                               idx + 2*n_regions], dtype=int)
        else:
            self.nbt = np.array([n_energy_in])
            self.interp = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2 * n_regions + 1
        self.energy_in = array[idx:idx + n_energy_in]

        # Location of distributions
        idx += n_energy_in
        loc_dist = np.asarray(array[idx:idx + n_energy_in], dtype=int)

        # Initialize variables
        self.intt = np.zeros(n_energy_in, dtype=int)
        self.num_discrete_lines = np.zeros(n_energy_in, dtype=int)
        self.energy_out = []
        self.pdf = []
        self.cdf = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin)
            INTTp = int(array[idx])
            if INTTp > 10:
                self.intt[i] = INTTp % 10
                self.num_discrete_lines[i] = (INTTp - self.intt[i])//10
            else:
                self.intt[i] = INTTp
                self.num_discrete_lines[i] = 0
            if self.intt[i] not in (1, 2):
                warn("Interpolation code on law 4 distribution is not 1 or 2.")

            n_energy_out = int(array[idx + 1])
            data = array[idx + 2:idx + 2 + 3*n_energy_out]
            data.shape = (3, n_energy_out)
            self.energy_out.append(data[0])
            self.pdf.append(data[1])
            self.cdf.append(data[2])


class Law5(EnergyDistribution):
    """General evaporation spectrum (From ENDF-6 FILE 5 LF=5)"""

    def __init__(self):
        pass

    def read(self, array, idx):
        # Read nuclear temperature as Tab1
        self.t = Tab1.from_ndarray(array, idx)

        # X-function
        nr = int(array[idx])
        ne = int(array[idx + 1 + 2*nr])
        idx += 2 + 2*nr + 2*ne
        net = int(array[idx])
        self.x = array[idx + 1:idx + 1 + net]


class Law7(EnergyDistribution):
    """Maxwell fission spectrum (From ENDF-6 File 5 LF=7)"""

    def __init__(self):
        pass

    def read(self, array, idx):
        # Read nuclear temperature as Tab1
        self.t = Tab1.from_ndarray(array, idx)

        # Restriction
        nr = int(array[idx])
        ne = int(array[idx + 1 + 2*nr])
        self.u = array[idx + 2 + 2*nr + 2*ne]


class Law9(EnergyDistribution):
    """Evaporation spectrum (From ENDF-6 File 5 LF=9)"""

    def __init__(self):
        pass

    def read(self, array, idx):
        # Read nuclear temperature as Tab1
        self.t = Tab1.from_ndarray(array, idx)

        # Restriction
        nr = int(array[idx])
        ne = int(array[idx + 1 + 2*nr])
        self.u = array[idx + 2 + 2*nr + 2*ne]


class Law11(EnergyDistribution):
    """Watt fission spectrum (From ENDF-6 File 5 LF=11)"""

    def __init__(self):
        pass

    def read(self, array, idx):
        # Energy-dependent a parameter
        self.a = Tab1.from_ndarray(array, idx)

        # Advance index
        nr = int(array[idx])
        ne = int(array[idx + 1 + 2*nr])
        idx += 2 + 2*nr + 2*ne

        # Energy-dependent b parameter
        self.b = Tab1.from_ndarray(array, idx)

        # Advance index
        nr = int(array[idx])
        ne = int(array[idx + 1 + 2*nr])
        idx += 2 + 2*nr + 2*ne

        # Rejection energy
        self.u = array[idx]


class Law44(EnergyDistribution):
    """Kalbach-87 formalism (ENDF-6 File 6 Law 1, LANG=2)"""

    def __init__(self):
        pass

    def read(self, array, idx, ldis):
        # Read number of interpolation regions and incoming energies
        n_regions = int(array[idx])
        n_energy_in = int(array[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            self.nbt = np.asarray(array[idx:idx + n_regions], dtype=int)
            self.interp = np.asarray(array[idx + n_regions:
                                               idx + 2*n_regions], dtype=int)
        else:
            self.nbt = np.array([n_energy_in])
            self.interp = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2 * n_regions + 1
        self.energy_in = array[idx:idx + n_energy_in]

        # Location of distributions
        idx += n_energy_in
        loc_dist = np.asarray(array[idx:idx + n_energy_in], dtype=int)

        # Initialize variables
        self.intt = np.zeros(n_energy_in, dtype=int)
        self.num_discrete_lines = np.zeros(n_energy_in, dtype=int)
        self.energy_out = []
        self.pdf = []
        self.cdf = []
        self.km_r = []
        self.km_a = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin)
            INTTp = int(array[idx])
            if INTTp > 10:
                self.intt[i] = INTTp % 10
                self.num_discrete_lines[i] = (INTTp - self.intt[i])//10
            else:
                self.intt[i] = INTTp
                self.num_discrete_lines[i] = 0
            if self.intt[i] not in (1, 2):
                warn("Interpolation code on law 4 distribution is not 1 or 2.")

            n_energy_out = int(array[idx + 1])
            data = array[idx + 2:idx + 2 + 5*n_energy_out]
            data.shape = (5, n_energy_out)
            self.energy_out.append(data[0])
            self.pdf.append(data[1])
            self.cdf.append(data[2])
            self.km_r.append(data[3])
            self.km_a.append(data[4])


class Law61(EnergyDistribution):
    """Tabular correlated energy-angular distribution"""

    def __init__(self):
        pass

    def read(self, array, idx, ldis):
        # Read number of interpolation regions and incoming energies
        n_regions = int(array[idx])
        n_energy_in = int(array[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            self.nbt = np.asarray(array[idx:idx + n_regions], dtype=int)
            self.interp = np.asarray(array[idx + n_regions:
                                               idx + 2*n_regions], dtype=int)
        else:
            self.nbt = np.array([n_energy_in])
            self.interp = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2 * n_regions + 1
        self.energy_in = array[idx:idx + n_energy_in]

        # Location of distributions
        idx += n_energy_in
        loc_dist = np.asarray(array[idx:idx + n_energy_in], dtype=int)

        # Initialize variables
        self.intt = np.zeros(n_energy_in, dtype=int)
        self.num_discrete_lines = np.zeros(n_energy_in, dtype=int)
        self.energy_out = []
        self.pdf = []
        self.cdf = []
        self.cosine_intt = []
        self.cosine_out = []
        self.cosine_pdf = []
        self.cosine_cdf = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intt = interpolation scheme (1=hist, 2=lin-lin)
            INTTp = int(array[idx])
            if INTTp > 10:
                self.intt[i] = INTTp % 10
                self.num_discrete_lines[i] = (INTTp - self.intt[i])//10
            else:
                self.intt[i] = INTTp
                self.num_discrete_lines[i] = 0
            if self.intt[i] not in (1, 2):
                warn("Interpolation code on law 4 distribution is not 1 or 2.")

            # Secondary energy distribution
            n_energy_out = int(array[idx + 1])
            data = array[idx + 2:idx + 2 + 4*n_energy_out]
            data.shape = (4, n_energy_out)
            self.energy_out.append(data[0])
            self.pdf.append(data[1])
            self.cdf.append(data[2])

            lc = np.asarray(data[3], dtype=int)

            # Secondary angular distributions
            self.cosine_intt.append(np.zeros(n_energy_out, dtype=int))
            self.cosine_out.append([])
            self.cosine_pdf.append([])
            self.cosine_cdf.append([])
            for j in range(n_energy_out):
                if lc[j] > 0:
                    idx = ldis + abs(lc[j]) - 1

                    self.cosine_intt[-1][j] = int(array[idx])
                    n_cosine = int(array[idx + 1])
                    data = array[idx + 2:idx + 2 + 3*n_cosine]
                    data.shape = (3, n_cosine)
                    self.cosine_out[-1].append(data[0])
                    self.cosine_pdf[-1].append(data[1])
                    self.cosine_cdf[-1].append(data[2])
                else:
                    # Isotropic distribution
                    self.cosine_intt[-1][j] = 1
                    self.cosine_out[-1].append(np.array([-1.0, 1.0]))
                    self.cosine_pdf[-1].append(np.array([0.5, 0.5]))
                    self.cosine_cdf[-1].append(np.array([0.0, 1.0]))


class Law66(EnergyDistribution):
    """N-body phase space distribution (ENDF File 6 Law 6)"""

    def __init__(self):
        pass

    def read(self, array, idx):
        self.n_bodies = int(array[idx])
        self.total_mass = array[idx + 1]


class Law67(EnergyDistribution):
    def __init__(self):
        pass

    def read(self, array, idx, ldis):
        # Read number of interpolation regions and incoming energies
        n_regions = int(array[idx])
        n_energy_in = int(array[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            self.nbt = np.asarray(array[idx:idx + n_regions], dtype=int)
            self.interp = np.asarray(array[idx + n_regions:
                                               idx + 2*n_regions], dtype=int)
        else:
            self.nbt = np.array([n_energy_in])
            self.interp = np.array([2])

        # Incoming energies at which distributions exist
        idx += 2 * n_regions + 1
        self.energy_in = array[idx:idx + n_energy_in]

        # Location of distributions
        idx += n_energy_in
        loc_dist = np.asarray(array[idx:idx + n_energy_in], dtype=int)

        # Initialize variables
        self.intmu = np.zeros(n_energy_in, dtype=int)
        self.intep = []
        self.cosine_out = []
        self.energy_out = []
        self.energy_out_pdf = []
        self.energy_out_cdf = []

        # Read each outgoing energy distribution
        for i in range(n_energy_in):
            idx = ldis + loc_dist[i] - 1

            # intmu = interpolation scheme (1=hist, 2=lin-lin)
            self.intmu[i] = int(array[idx])

            # Number of equiprobable cosines
            n_cosine = int(array[idx + 1])

            # Secondary cosines
            idx += 2
            self.cosine_out.append(array[idx:idx + n_cosine])

            # locations of energy distributions
            idx += n_cosine
            lmu = np.asarray(array[idx:idx + n_cosine])

            # Secondary energy distribution
            self.intep.append(np.zeros(n_cosine, dtype=int))
            self.energy_out.append([])
            self.energy_out_pdf.append([])
            self.energy_out_cdf.append([])

            for j in range(n_cosine):
                idx = ldis + lmu[j] - 1

                self.intep[-1][j] = int(array[idx])
                n_energy_out = int(array[idx + 1])
                data = array[idx + 2:idx + 2 + 3*n_energy_out]
                data.shape = (3, n_energy_out)
                self.energy_out[-1].append(data[0])
                self.energy_out_pdf[-1].append(data[1])
                self.energy_out_cdf[-1].append(data[2])


class SabTable(AceTable):
    """A SabTable object contains thermal scattering data as represented by
    an S(alpha, beta) table.

    Parameters
    ----------
    name : str
        ZAID identifier of the table, e.g. lwtr.10t.
    awr : float
        Atomic mass ratio of the target nuclide.
    temp : float
        Temperature of the target nuclide in eV.

    :Attributes:
      **awr** : float
        Atomic mass ratio of the target nuclide.

      **elastic_e_in** : list of floats
        Incoming energies in MeV for which the elastic cross section is
        tabulated.

      **elastic_P** : list of floats
        Elastic scattering cross section for data derived in the incoherent
        approximation, or Bragg edge parameters for data derived in the coherent
        approximation.

      **elastic_type** : str
        Describes the behavior of the elastic cross section, i.e. whether it was
        derived in the incoherent or coherent approximation.

      **inelastic_e_in** : list of floats
        Incoming energies in MeV for which the inelastic cross section is
        tabulated.

      **inelastic_sigma** : list of floats
        Inelastic scattering cross section in barns at each energy.

      **name** : str
        ZAID identifier of the table, e.g. 92235.70c.

      **temp** : float
        Temperature of the target nuclide in eV.

    """
    

    def __init__(self, name, awr, temp):
        super(SabTable, self).__init__(name, awr, temp)

    def _read_all(self):
        self._read_itie()
        self._read_itce()
        self._read_itxe()
        self._read_itca()

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Thermal S(a,b) Table: {0}>".format(self.name)
        else:
            return "<ACE Thermal S(a,b) Table>"

    def _read_itie(self):
        """Read energy-dependent inelastic scattering cross sections.
        """
        idx = self.jxs[1]
        NE = int(self.xss[idx])
        self.inelastic_e_in = self.xss[idx+1:idx+1+NE]
        self.inelastic_sigma = self.xss[idx+1+NE:idx+1+2*NE]

    def _read_itce(self):
        """Read energy-dependent elastic scattering cross sections.
        """
        # Determine if ITCE block exists
        idx = self.jxs[4]
        if idx == 0:
            return

        # Read values
        NE = int(self.xss[idx])
        self.elastic_e_in = self.xss[idx+1:idx+1+NE]
        self.elastic_P = self.xss[idx+1+NE:idx+1+2*NE]

        if self.nxs[5] == 4:
            self.elastic_type = 'sigma=P'
        else:
            self.elastic_type = 'sigma=P/E'

    def _read_itxe(self):
        """Read coupled energy/angle distributions for inelastic scattering.
        """
        # Determine number of energies and angles
        NE_in = len(self.inelastic_e_in)
        NE_out = self.nxs[4]
        NMU = self.nxs[3]
        idx = self.jxs[3]
        
        self.inelastic_e_out = self.xss[idx:idx+NE_in*NE_out*(NMU+2):NMU+2]
        self.inelastic_e_out.shape = (NE_in, NE_out)

        self.inelastic_mu_out = self.xss[idx:idx+NE_in*NE_out*(NMU+2)]
        self.inelastic_mu_out.shape = (NE_in, NE_out, NMU+2)
        self.inelastic_mu_out = self.inelastic_mu_out[:,:,1:]

    def _read_itca(self):
        """Read angular distributions for elastic scattering.
        """
        NMU = self.nxs[6]
        if self.jxs[4] == 0 or NMU == -1:
            return
        idx = self.jxs[6]

        NE = len(self.elastic_e_in)
        self.elastic_mu_out = self.xss[idx:idx+NE*NMU]
        self.elastic_mu_out.shape = (NE, NMU)

            
class Reaction(object):
    """Reaction(MT, table=None)

    A Reaction object represents a single reaction channel for a nuclide with
    an associated cross section and, if present, a secondary angle and energy
    distribution. These objects are stored within the ``reactions`` attribute on
    subclasses of AceTable, e.g. NeutronTable.

    Parameters
    ----------
    MT : int
        The ENDF MT number for this reaction. On occasion, MCNP uses MT numbers
        that don't correspond exactly to the ENDF specification.
    table : AceTable
        The ACE table which contains this reaction. This is useful if data on
        the parent nuclide is needed (for instance, the energy grid at which
        cross sections are tabulated)

    :Attributes:
      **ang_energy_in** : list of floats
        Incoming energies in MeV at which angular distributions are tabulated.

      **ang_energy_cos** : list of floats
        Scattering cosines corresponding to each point of the angular distribution
        functions.

      **ang_energy_pdf** : list of floats
        Probability distribution function for angular distribution.

      **ang_energy_cdf** : list of floats
        Cumulative distribution function for angular distribution.

      **e_dist_energy** : list of floats
        Incoming energies in MeV at which energy distributions are tabulated.

      **e_dist_law** : int
        ACE law used for secondary energy distribution.

      **IE** : int
        The index on the energy grid corresponding to the threshold of this
        reaction.

      **MT** : int
        The ENDF MT number for this reaction. On occasion, MCNP uses MT numbers
        that don't correspond exactly to the ENDF specification.

      **Q** : float
        The Q-value of this reaction in MeV.

      **sigma** : list of floats
        Microscopic cross section for this reaction at each point on the energy
        grid above the threshold value.

      **TY** : int
        An integer whose absolute value is the number of neutrons emitted in
        this reaction. If negative, it indicates that scattering should be
        performed in the center-of-mass system. If positive, scattering should
        be preformed in the laboratory system.

    """

    def __init__(self, MT, table=None):
        self.table = table # Reference to containing table
        self.MT = MT       # MT value
        self.Q = None      # Q-value
        self.TY = None     # Neutron release
        self.IE = 1        # Energy grid index
        self.sigma = []    # Cross section values

    def broaden(self, T_high):
        pass        

    def threshold(self):
        """threshold()

        Return energy threshold for this reaction.
        """
        return self.table.energy[self.IE - 1]

    def __repr__(self):
        try:
            name = label(self.MT)
        except ValueError:
            # Occurs for photon reactions with MTs like 4001, etc.
            name = None

        if name is not None:
            rep = "<ACE Reaction: MT={0} {1}>".format(self.MT, name)
        else:
            rep = "<ACE Reaction: Unknown MT={0}>".format(self.MT)
        return rep


class DosimetryTable(AceTable):

    def __init__(self, name, awr, temp):
        super(DosimetryTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Dosimetry Table: {0}>".format(self.name)
        else:
            return "<ACE Dosimetry Table>"
        

class NeutronDiscreteTable(AceTable):

    def __init__(self, name, awr, temp):
        super(NeutronDiscreteTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Discrete-E Neutron Table: {0}>".format(self.name)
        else:
            return "<ACE Discrete-E Neutron Table>"
        

class NeutronMGTable(AceTable):

    def __init__(self, name, awr, temp):
        super(NeutronMGTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Multigroup Neutron Table: {0}>".format(self.name)
        else:
            return "<ACE Multigroup Neutron Table>"
        

class PhotoatomicTable(AceTable):

    def __init__(self, name, awr, temp):
        super(PhotoatomicTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Continuous-E Photoatomic Table: {0}>".format(self.name)
        else:
            return "<ACE Continuous-E Photoatomic Table>"
        

class PhotoatomicMGTable(AceTable):

    def __init__(self, name, awr, temp):
        super(PhotoatomicMGTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Multigroup Photoatomic Table: {0}>".format(self.name)
        else:
            return "<ACE Multigroup Photoatomic Table>"
        

class ElectronTable(AceTable):

    def __init__(self, name, awr, temp):
        super(ElectronTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Electron Table: {0}>".format(self.name)
        else:
            return "<ACE Electron Table>"
        

class PhotonuclearTable(AceTable):

    def __init__(self, name, awr, temp):
        super(PhotonuclearTable, self).__init__(name, awr, temp)

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<ACE Photonuclear Table: {0}>".format(self.name)
        else:
            return "<ACE Photonuclear Table>"

table_types = {
    "c": NeutronTable,
    "t": SabTable,
    "y": DosimetryTable,
    "d": NeutronDiscreteTable,
    "p": PhotoatomicTable,
    "m": NeutronMGTable,
    "g": PhotoatomicMGTable,
    "e": ElectronTable,
    "u": PhotonuclearTable}

_distributions = {1: Law1, 2: Law2, 3: Law3, 4: Law4, 5: Law5, 7: Law7,
                  9: Law9, 11: Law11, 44: Law44, 61: Law61, 66: Law66,
                  67: Law67}

if __name__ == '__main__':
    # Might be nice to check environment variable DATAPATH to search for xsdir
    # and list files that could be read?
    pass
