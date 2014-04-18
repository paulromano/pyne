#!/usr/bin/env python

"""Module for parsing and manipulating data from ENDF evaluations. Currently, it
only can read several MTs from File 1, but with time it will be expanded to
include the entire ENDF format.

All the classes and functions in this module are based on document
ENDF-102 titled "Data Formats and Procedures for the Evaluated Nuclear
Data File ENDF-6". The latest version from June 2009 can be found at
http://www-nds.iaea.org/ndspub/documents/endf/endf102/endf102.pdf

For more information on the Evaluation class, contact Paul Romano
<paul.k.romano@gmail.com>. For more information on the Library class, contact
John Xia <john.danger.xia@gmail.com>.
"""

from __future__ import print_function, division, unicode_literals
import re
import os
from collections import OrderedDict
from libc.stdlib cimport malloc, free

cimport numpy as np
import numpy as np
from scipy.interpolate import interp1d

np.import_array()

from pyne cimport cpp_nucname
from pyne import nucname
import pyne.rxdata as rx
from pyne.rxname import label
from pyne.utils import fromendf_tok, endftod

from libc.stdlib cimport atof, atoi
from libc.string cimport strtok, strcpy, strncpy

libraries = {0: "ENDF/B", 1: "ENDF/A", 2: "JEFF", 3: "EFF",
             4: "ENDF/B High Energy", 5: "CENDL", 6: "JENDL",
             31: "INDL/V", 32: "INDL/A", 33: "FENDL", 34: "IRDF",
             35: "BROND", 36: "INGDB-90", 37: "FENDL/A", 41: "BROND"}
radiation_type = {0: 'gamma', 1: 'beta-', 2: 'ec/beta+', 3: 'IT',
                  4: 'alpha', 5: 'neutron', 6: 'sf', 7: 'proton',
                  8: 'e-', 9: 'xray', 10: 'unknown'}
resonance_types = {1: 'SLBW', 2: 'MLBW', 3: 'RM', 4: 'AA', 7: 'RML'}
FILE1_R = re.compile(r'1451 *\d{1,5}$')
CONTENTS_R = re.compile(' +\d{1,2} +\d{1,3} +\d{1,10} +')
SPACE66_R = re.compile(' {66}')
NUMERICAL_DATA_R = re.compile('[\d\-+. ]{80}\n$')
SPACE66_R = re.compile(' {66}')

class Library(rx.RxLib):
    """A class for a file which contains multiple ENDF evaluations."""
    def __init__(self, fh):
        self.mts = {}
        self.structure = {}
        self.mat_dict = {}
        self.more_files = True
        self.intdict = {1: self._histogram, 2: self._linlin, 3: self._linlog, 4:
                        self._loglin, 5: self._loglog, 6:self._chargedparticles,
                        11: self._histogram, 12: self._linlin, 13: self._linlog,
                        14: self._loglin, 15: self._loglog, 21: self._histogram,
                        22: self._linlin, 23: self._linlog, 24: self._loglin,
                        25: self._loglog}
        self.chars_til_now = 0
        self.offset = 0
        self.fh = fh
        while self.more_files:
            self._read_headers()


    def load(self):
        """load()
        Read the ENDF file into a NumPy array.

        Returns
        --------
        data : np.array, 1d, float64
            Returns a 1d float64 NumPy array.
        """
        opened_here = False
        if isinstance(self.fh, basestring):
            fh = open(self.fh, 'r')
            opened_here = True
        else:
            fh = self.fh
        fh.readline()
        data = fromendf_tok(fh.read())
        fh.seek(0)
        if opened_here:
            fh.close()
        return data

    def _read_headers(self):
        cdef int nuc
        cdef int mat_id
        cdef double nucd
        opened_here = False
        if isinstance(self.fh, basestring):
            fh = open(self.fh, 'r')
            opened_here = True
        else:
            fh = self.fh
        # Skip the first line and get the material ID.
        fh.seek(self.chars_til_now)
        len_headline = len(fh.readline())
        self.offset += 81 - len_headline
        line = fh.readline()
        mat_id = int(line[66:70].strip() or -1)
        # originally in a float version of ZZAAA.M, ie 94242.1
        nuc = cpp_nucname.id(<int> (endftod(line[:11])*10))
        # Make a new dict in self.structure to contain the material data.
        if nuc not in self.structure:
            self.structure.update(
                {nuc:{'styles': "", 'docs': [], 'particles': [], 'data': {},
                         'matflags': {}}})
            self.mat_dict.update({nuc:{'end_line':[],
                                          'mfs':{}}})
        # Parse header (all lines with 1451)
        mf = 1
        stop = (self.chars_til_now+self.offset)//81
        while FILE1_R.search(line):
            # parse contents section
            if CONTENTS_R.match(line):
                # When MF and MT change, add offset due to SEND/FEND records.
                old_mf = mf
                mf, mt = int(line[22:33]), int(line[33:44])
                mt_length = int(line[44:55])
                if old_mf == mf:
                    start = stop + 1
                else:
                    start = stop + 2
                stop = start + mt_length
                self.mat_dict[nuc]['mfs'][mf,mt] = (81*start-self.offset,
                                                    81*stop-self.offset)
                line = fh.readline()
            # parse comment
            elif SPACE66_R.match(line):
                self.structure[nuc]['docs'].append(line[0:66])
                line = fh.readline()
            elif NUMERICAL_DATA_R.match(line):
                line = fh.readline()
                continue
            else:
                self.structure[nuc]['docs'].append(line[0:66])
                line = fh.readline()
        # Find where the end of the material is and then jump to it.
        self.chars_til_now = (stop + 4)*81 - self.offset
        fh.seek(self.chars_til_now)
        nextline = fh.readline()
        self.more_files = (nextline != '' and nextline[68:70] != "-1")
        # Update materials dict
        if mat_id != -1:
            self.mat_dict[nuc]['end_line'] = (self.chars_til_now+self.offset)//81
            setattr(self, "mat{0}".format(nuc), self.structure[nuc])
        self._read_mat_flags(nuc)
        fh.seek(0)
        if opened_here:
            fh.close()

    def _read_mat_flags(self, nuc):
        """Reads the global flags for a certain material.

        Parameters
        -----------
        nuc: int
            ZZAAAM of material.
        """
        mf1 = self.get_rx(nuc, 1, 451, lines=4)
        flagkeys = ['ZA', 'AWR', 'LRP', 'LFI', 'NLIB', 'NMOD', 'ELIS',
                    'STA', 'LIS', 'LIS0', 0, 'NFOR', 'AWI', 'EMAX',
                    'LREL', 0, 'NSUB', 'NVER', 'TEMP', 0, 'LDRV',
                    0, 'NWD', 'NXC']
        flags = dict(zip(flagkeys, mf1[:12]))
        del flags[0]
        self.structure[nuc]['matflags'] = flags

    def _get_cont(self, keys, line):
        """Read one line of the array, treating it as a CONT record.

        Parameters
        -----------
        keys: iterable
            An iterable containing the labels for each field in the CONT record.
            For empty/unassigned fields, use 0.
        line: array-like
            The line to be read.

        Returns
        --------
        cont : dict
            Contains labels and values mapped to each other.
        """
        cont = dict(zip(keys, line.flat[:6]))
        if 0 in cont:
            del cont[0]
        return cont

    def _get_head(self, keys, line):
        """Read one line of the array, treating it as a HEAD record.

        Parameters
        -----------
        keys: iterable
            An iterable containing the labels for each field in the HEAD record.
            For empty/unassigned fields, use 0.
        line: array-like
            The line to be read.

        Returns
        --------
        cont : dict
            Contains labels and values mapped to each other.
        """
        # Just calls self._get_cont because HEAD is just a special case of CONT
        if (keys[0] == 'ZA' and keys[1] == 'AWR'):
            return self._get_cont(keys, line)
        else:
            raise ValueError('This is not a HEAD record: {}'.format(
                    dict(zip(keys,line))))

    def _get_list(self, headkeys, itemkeys, lines):
        """Read some lines of the array, treating it as a LIST record.

        Parameters
        -----------
        headkeys: iterable
            An iterable containing the labels for each field in the first
            record. For empty/unassigned fields, use 0.
        itemkeys: iterable
            An iterable containing the labels for each field in the next
            records. For empty/unassigned fields, use 0. If itemkeys has length
            1, the array is flattened and assigned to that key.
        lines: two-dimensional array-like
            The lines to be read. Each line should have 6 elements. The first
            line should be the first line of the LIST record; since we don't
            know the length of the LIST record, the last line should be the last
            line it is plausible for the LIST record to end.

        Returns
        --------
        head: dict
            Contains elements of the first line paired with their labels.
        items: dict
            Contains columns of the LIST array paired with their labels, unless
            itemkeys has length 1, in which case items contains the flattened
            LIST array paired with its label.
        total_lines: int
            The number of lines the LIST record takes up.
        """

        head = dict(zip(headkeys, lines[0:].flat[:len(headkeys)]))
        if 0 in head:
            del head[0]
        npl = int(lines[0][4])
        headlines = (len(headkeys)-1)//6 + 1
        arraylines = (npl-1)//6 + 1
        if len(itemkeys) == 1:
            array_len = npl - (headlines-1) * 6
            items={itemkeys[0]: lines[headlines:].flat[:array_len]}
        else:
            array_width = ((len(itemkeys)-1)//6 + 1)*6
            items_transposed = np.transpose(
                lines[headlines:headlines+arraylines].reshape(-1,
                                                              array_width))
            items = dict(zip(itemkeys, items_transposed))
        if 0 in items:
            del items[0]

        total_lines = 1+arraylines
        return head, items, total_lines

    def _get_tab1(self, headkeys, xykeys,lines):
        """Read some lines of the array, treating it as a TAB1 record.

        Parameters
        -----------
        headkeys: iterable, length 6
            An iterable containing the labels for each field in the first
            line. For empty/unassigned fields, use 0.
        xykeys: iterable, length 2
            An iterable containing the labels for the interpolation data. The
            first key should be xint, the second should be y(x).
        lines: two-dimensional array-like
            The lines to be read. Each line should have 6 elements. The first
            line should be the first line of the TAB1 record; since we don't
            know the length of the TAB1 record, the last line should be the last
            line it is plausible for the TAB1 record to end.

        Returns
        --------
        head: dict
            Contains elements of the first card paired with their labels.
        intdata: dict
            Contains the interpolation data.
        total_lines: int
            The number of lines the TAB1 record takes up.
        """
        head = dict(zip(headkeys, lines[0]))
        if 0 in head:
            del head[0]
        nr, np_ = int(lines[0][4]), int(lines[0][5])
        meta_len = (nr*2-1)//6 + 1
        data_len = (np_*2-1)//6 + 1
        intmeta = dict(zip(('intpoints','intschemes'),
                           (lines[1:1+meta_len].flat[:nr*2:2],
                            lines[1:1+meta_len].flat[1:nr*2:2])))
        intdata = dict(zip(xykeys,
            (lines[1+meta_len:1+meta_len+data_len].flat[:np_*2:2],
             lines[1+meta_len:1+meta_len+data_len].flat[1:np_*2:2])))
        intdata.update(intmeta)
        total_lines = 1 + meta_len + data_len
        return head, intdata, total_lines

    def _histogram(self, e_int, xs, low, high):
        if low in e_int:
            # truncate at lower bound
            xs = xs[e_int >= low]
            e_int = e_int[e_int >= low]
        elif low is not None and low > e_int[0]:
            # truncate at lower bound and prepend interpolated endpoint
            low_xs = xs[e_int < low][-1]
            xs = np.insert(xs[e_int > low], 0, low_xs)
            e_int = np.insert(e_int[e_int > low], 0, low)
        if high in e_int:
            # truncate at higher bound
            xs = xs[e_int <= high]
            e_int = e_int[e_int <= high]
        elif high is not None:
            # truncate at higher bound and prepend interpolated endpoint
            high_xs = xs[e_int < high][-1]
            xs = np.append(xs[e_int < high], high_xs)
            e_int = np.append(e_int[e_int < high], high)
        de_int = float(e_int[-1]-e_int[0])
        return np.nansum((e_int[1:]-e_int[:-1]) * xs[:-1]/de_int)

    def _linlin(self, e_int, xs, low, high):
        if low is not None or high is not None:
            interp = interp1d(e_int, xs)
            if low in e_int:
                xs = xs[e_int >= low]
                e_int = e_int[e_int >= low]
            elif low is not None and low > e_int[0]:
                low_xs = interp(low)
                xs = np.insert(xs[e_int > low], 0, low_xs)
                e_int = np.insert(e_int[e_int > low], 0, low)
            if high in e_int:
                xs = xs[e_int <= high]
                e_int = e_int[e_int <= high]
            elif high is not None:
                high_xs = interp(high)
                xs = np.append(xs[e_int < high], high_xs)
                e_int = np.append(e_int[e_int < high], high)
        de_int = float(e_int[-1]-e_int[0])
        return np.nansum((e_int[1:]-e_int[:-1])* (xs[1:]+xs[:-1])/2./de_int)

    def _linlog(self, e_int, xs, low, high):
        if low is not None or high is not None:
            interp = interp1d(np.log(e_int), xs)
            if low in e_int:
                xs = xs[e_int >= low]
                e_int = e_int[e_int >= low]
            elif low is not None and low > e_int[0]:
                low_xs = interp(np.log(low))
                xs = np.insert(xs[e_int > low], 0, low_xs)
                e_int = np.insert(e_int[e_int > low], 0, low)
            if high in e_int:
                xs = xs[e_int <= high]
                e_int = e_int[e_int <= high]
            elif high is not None:
                high_xs = interp(np.log(high))
                xs = np.append(xs[e_int < high], high_xs)
                e_int = np.append(e_int[e_int < high], high)

        de_int = float(e_int[-1]-e_int[0])
        x1 = e_int[:-1]
        x2 = e_int[1:]
        y1 = xs[:-1]
        y2 = xs[1:]
        A = (y1-y2)/(np.log(x1/x2))
        B = y1-A*np.log(x1)
        return np.nansum(A*(x2*np.log(x2) - x1*np.log(x1)-x2+x1) + B*(x2-x1))/de_int

    def _loglin(self, e_int, xs, low, high):
        if low is not None or high is not None:
            interp = interp1d(e_int, np.log(xs))
            if low in e_int:
                xs = xs[e_int >= low]
                e_int = e_int[e_int >= low]
            elif low is not None and low > e_int[0]:
                low_xs = np.e ** interp(low)
                xs = np.insert(xs[e_int > low], 0, low_xs)
                e_int = np.insert(e_int[e_int > low], 0, low)
            if high in e_int:
                xs = xs[e_int <= high]
                e_int = e_int[e_int <= high]
            elif high is not None:
                high_xs = np.e ** interp(high)
                xs = np.append(xs[e_int < high], high_xs)
                e_int = np.append(e_int[e_int < high], high)

        de_int = float(e_int[-1]-e_int[0])
        x1 = e_int[:-1]
        x2 = e_int[1:]
        y1 = xs[:-1]
        y2 = xs[1:]
        A = (np.log(y1)-np.log(y2))/(x1-x2)
        B = np.log(y1) - A*x1
        return np.nansum((y2-y1)/A)/de_int

    def _loglog(self, e_int, xs, low, high):
        if low is not None or high is not None:
            interp = interp1d(np.log(e_int), np.log(xs))
            if low in e_int:
                xs = xs[e_int >= low]
                e_int = e_int[e_int >= low]
            elif low is not None and low > e_int[0]:
                low_xs = np.e ** interp(np.log(low))
                xs = np.insert(xs[e_int > low], 0, low_xs)
                e_int = np.insert(e_int[e_int > low], 0, low)
            if high in e_int:
                xs = xs[e_int <= high]
                e_int = e_int[e_int <= high]
            elif high is not None:
                high_xs = np.e ** interp(np.log(high))
                xs = np.append(xs[e_int < high], high_xs)
                e_int = np.append(e_int[e_int < high], high)

        de_int = float(e_int[-1]-e_int[0])
        x1 = e_int[:-1]
        x2 = e_int[1:]
        y1 = xs[:-1]
        y2 = xs[1:]
        A = - np.log(y2/y1)/np.log(x1/x2)
        B = - (np.log(y1)*np.log(x2) - np.log(y2)*np.log(x1))/np.log(x1/x2)
        return np.nansum(np.e**B / (A+1) * (x2**(A+1) - x1**(A+1))/de_int)

    def _chargedparticles(self, e_int, xs, flags=None):
        q = flags['Q']
        if q > 0:
            T = 0
        else:
            T = q
        de_int = float(e_int[-1]-e_int[0])
        x1 = e_int[:-1]
        x2 = e_int[1:]
        y1 = xs[:-1]
        y2 = xs[1:]
        B = np.log(y2*x2/(x1*y1)) / (1/(x1-T)**0.5 - 1/(x2-T)**0.5)
        A = np.e**(B/(x1-T)**0.5)*y1*x1
        # FIXME
        raise NotImplementedError("see docs for more details.")

    def integrate_tab_range(self, intscheme, e_int, xs, low=None, high=None):
        """integrate_tab_range(intscheme, e_int, xs, low=None, high=None)
        Integrates across one tabulation range.

        Parameters
        ----------
        intscheme : int or float
            The interpolation scheme used in this range.
        e_int : array
            The energies at which we have xs data.
        xs : array
            The xs data corresponding to e_int.
        low, high : float
            Lower and upper bounds within the tabulation range to start/stop at.

        Returns
        -------
        sigma_g : float
            The group xs.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            # each of these functions returns a normalized integration
            # over the range
            return self.intdict[intscheme](e_int, xs, low, high)

    def _cont_and_update(self, flags, keys, data, total_lines):
        flags.update(self._get_cont(keys, data[total_lines]))
        return flags, total_lines+1

    def _nls_njs_loop(self, L_keys, j_keys, itemkeys, data, total_lines,
                     range_flags, subsection_dict):
        nls = int(range_flags['NLS'])
        for nls_iter in range(nls):
            if j_keys is None:
                L_flags, items, lines = self._get_list(
                    L_keys, itemkeys, data[total_lines:])
                total_lines += lines
                spi, L = range_flags['SPI'], L_flags['L']
                subsection_dict[spi, L] = items
            else:
                L_flags = self._get_cont(L_keys, data[total_lines])
                total_lines += 1
                njs = int(L_flags['NJS'])
                for njs_iter in range(njs):
                    j_flags, items, lines = self._get_list(
                        j_keys, itemkeys, data[total_lines:])
                    total_lines += lines
                    items.update(j_flags)
                    spi, L, aj = range_flags['SPI'], L_flags['L'], j_flags['AJ']
                    subsection_dict[(spi, L, aj)] = items
        return total_lines

    def _read_res(self, mat_id):
        """_read_res(mat_id)
        Read the resonance data from one material in the library and updates
        self.structure.

        Parameters
        -----------
        mat_id: int
            Material id .
        """
        lrp = self.structure[mat_id]['matflags']['LRP']
        if (lrp == -1 or mat_id in (-1,0)):
            # If the LRP flag for the material is -1, there's no resonance data.
            # Also if the mat id is invalid.
            pass
        else:
            # Load the resonance data.
            mf2 = self.get_rx(mat_id,2,151).reshape(-1, 6)

            self.structure[mat_id]['matflags'].update(
                self._get_head(['ZA','AWR',0,0,'NIS',0], mf2[0]))
            total_lines = 1
            for isotope_num in range(
                    int(self.structure[mat_id]['matflags']['NIS'])):
                total_lines += self._read_nis(mf2[total_lines:], lrp, mat_id)
        for isotope in self.structure[mat_id]['data'].values():
            isotope['resolved'].sort()
            isotope['unresolved'].sort()

    def _read_nis(self, isotope_data, lrp, mat_id):
        """_read_nis(isotope_data, lrp, mat_id)
        Read resonance data for a specific isotope.

        Parameters
        -----------
        isotope_data: 2D array
            The section of the resonance data to read. The isotope starts at the
            top of this.
        lrp: int
            A flag denoting the type of data in the isotope. Exact meaning of
            this flag can be found in ENDF Manual pp.50-51.
        mat_id: int
            Material ZZAAAM.

        Returns
        --------
        total_lines: int
            The number of lines the isotope takes up.
        """
        isotope_flags = self._get_cont(['ZAI','ABN',0,'LFW','NER',0],
                                       isotope_data[0])
        nuc_i = nucname.id(int(isotope_flags['ZAI']*10))
        self.structure[mat_id]['data'].update(
            {nuc_i:{'resolved':[],
                       'unresolved':[],
                       'datadocs':[],
                       'xs':{},
                       'output':{'channel1':[],
                                 'channel2':[]},
                       'isotope_flags': isotope_flags}})
        total_lines = 1
        for er in range(int(isotope_flags['NER'])):
            total_lines += self._read_subsection(isotope_data[total_lines:],
                                                 isotope_flags,
                                                 mat_id,
                                                 nuc_i)

        return total_lines

    def _read_subsection(self, subsection, isotope_flags, mat_id, nuc_i):
        """Read resonance data for a specific energy range subsection.

        Parameters
        -----------
        subsection: 2D array
            The section of the resonance data to read. The energy range
            subsection starts at the top of this.
        range_flags: dict
            Dictionary of flags inherited from the range.
        isotope_flags: dict
            Dictionary of flags inherited from the isotope.
        mat_id: int
            Material ZZAAAM.
        nuc_i: int
            Isotope ZZAAAM.

        Returns
        --------
        total_lines: int
            The number of lines the energy range subsection takes up.
        """
        range_flags = self._get_cont(('EL','EH','LRU','LRF','NRO','NAPS'),
                                     subsection[0])
        total_lines = 1
        lru = int(round(range_flags['LRU']))
        lru_list = [self._read_ap_only, self._read_resolved,
                    self._read_unresolved]
        total_lines += lru_list[lru](subsection[1:],
                                     range_flags,
                                     isotope_flags,
                                     mat_id,
                                     nuc_i)
        return total_lines

    def _read_resolved(self, subsection, range_flags, isotope_flags, mat_id,
                       nuc_i):
        """Read the subsection for a resolved energy range.

        Parameters
        -----------
        subsection: 2D array
            The section of the resonance data to read. The energy range
            subsection starts at the top of this.
        range_flags: dict
            Dictionary of flags inherited from the range.
        isotope_flags: dict
            Dictionary of flags inherited from the isotope.
        mat_id: int
            ZZAAAM of the material.
        nuc_i: int
            ZZAAAM of the isotope.

        Returns
        --------
        total_lines: int
            The number of lines taken up by the subsection.
        """
        def read_kbks(nch, subsection, aj_data, total_lines):
            for ch in range(nch):
                lbk = int(subsection[total_lines][4])
                lbk_list_keys = {2: ('R0','R1','R2','S0','S1',0),
                                 3: ('R0','SO','GA',0,0,0)}
                aj_data['ch{}'.format(ch)] = {'LBK': lbk}
                ch_data = aj_data['ch{}'.format(ch)]
                if lbk == 0:
                    total_lines += 2
                elif lbk == 1:
                    total_lines += 2
                    rbr, rbr_size = self._get_tab1(
                        (0,0,0,0,'NR','NP'), ('e_int','RBR'),
                        subsection[total_lines:])[1:3]
                    total_lines += rbr_size
                    ch_data['RBR'] = rbr
                    rbi, rbi_size = self._get_tab1(
                        (0,0,0,0,'NR','NP'), ('e_int','RBI'),
                        (subsection[total_lines:]))[1:3]
                    total_lines += rbi_size
                    ch_data['RBI'] = rbi
                else:
                    ch_data, total_lines = self._cont_and_update(
                        ch_data, ('ED','EU',0,0,'LBK',0), subsection,
                        total_lines)
                    ch_data, total_lines = self._cont_and_update(
                        ch_data, lbk_list_keys[lbk], subsection,
                        total_lines)
            return total_lines

        def read_kpss(nch, subsection, aj_data, total_lines):
            for ch in range(nch):
                ch_data = aj_data['ch{}'.format(ch)]
                lps = subsection[total_lines][4]
                ch_data['LPS'] = lps
                total_lines += 2
                if lps == 1:
                    psr, psr_size = self._get_tab1(
                        (0,0,0,0,'NR','NP'), ('e_int','PSR'),
                        subsection[total_lines:])[1:3]
                    total_lines += psr_size
                    ch_data['PSR'] = psr
                    psi, psi_size = self._get_tab1(
                        (0,0,0,0,'NR','NP'), ('e_int','PSI'),
                        (subsection[total_lines:]))[1:3]
                    total_lines += psi_size
                    ch_data['PSI'] = psi
                    total_lines += psi_size
            return total_lines

        lrf = int(range_flags['LRF'])
        subsection_dict = rx.DoubleSpinDict({})
        headers = [None,
                   ('SPI','AP',0,0,'NLS',0),
                   ('SPI','AP',0,0,'NLS',0),
                   ('SPI','AP','LAD',0,'NLS','NLSC'),
                   ('SPI','AP',0,0,'NLS',0),
                   None,
                   None,
                   (0,0,'IFG','KRM','NJS','KRL')]
        if range_flags['NRO'] > 0:
            intdata, total_lines = self._get_tab1((0,0,0,0,'NR','NP'),
                                                  ('E','AP'),
                                                  subsection)[1:3]
            subsection_dict['int'] = intdata
        else:
            total_lines = 0
        range_flags, total_lines = self._cont_and_update(
                range_flags, headers[lrf], subsection, total_lines)

        lrf_L_keys = [None,
                      ('AWRI','QX','L','LRX','6*NRS','NRS'),
                      ('AWRI','QX','L','LRX','6*NRS','NRS'),
                      ('AWRI','APL','L',0,'6*NRS','NRS'),
                      (0,0,'L',0,'NJS',0)]
        lrf_J_keys = [None, None, None, None, ('AJ',0,0,0,'12*NLJ','NLJ')]
        lrf_itemkeys = [None,
                        ('ER','AJ','GT','GN','GG','GF'),
                        ('ER','AJ','GT','GN','GG','GF'),
                        ('ER','AJ','GN','GG','GFA','GFB'),
                        ('DET','DWT','GRT','GIT','DEF','DWF','GRF','GIF','DEC',
                         'DWC','GRC','GIC')]
        if lrf == 4:
            # Adler-Adler
            bg_flags, bg, bg_size = self._get_list(
                ('AWRI',0,'LI',0,'6*NX','NX'),
                ('A1','A2','A3','A4','B1','B2'),
                subsection[total_lines:])
            total_lines += bg_size
            subsection_dict['bg'] = bg

        if lrf < 5:
            total_lines = self._nls_njs_loop(lrf_L_keys[lrf],
                                            lrf_J_keys[lrf],
                                            lrf_itemkeys[lrf],
                                            subsection,
                                            total_lines,
                                            range_flags,
                                            subsection_dict)
        if lrf == 7:
            # R-Matrix Limited Format (ENDF Manual pp. 62-67)
            # Particle pair descriptions for the whole range
            particle_pair_data, pp_size = self._get_list(
                (0,0,'NPP',0,'12*NPP','2*NPP'),
                ('MA','MB','ZA','ZB','IA','IB','Q','PNT','SHF','MT','PA','PB'),
                subsection[total_lines:])[1:3]
            total_lines += pp_size
            range_flags.update(particle_pair_data)
            for aj_section in range(int(range_flags['NJS'])):
                # Read first LIST record, with channel descriptions
                aj_flags, ch_items, ch_size = self._get_list(
                    ('AJ','PJ','KBK','KPS','6*NCH','NCH'),
                    ('IPP','L','SCH','BND','APE','APT'),
                    subsection[total_lines:])
                total_lines += ch_size
                # Second LIST record, with resonance energies and widths.
                er_flags, er_data, er_size = self._get_list(
                    (0,0,0,'NRS','6*NX','NX'), ('ER',), subsection[total_lines:])
                total_lines += er_size
                nch = int(aj_flags['NCH'])
                er_array_width = (nch//6+1)*6
                er_data = er_data['ER'].reshape(-1,er_array_width).transpose()
                aj_data = {'ER': er_data[0], 'GAM': er_data[1:1+nch].transpose()}
                aj_data.update(ch_items)
                aj = aj_flags['AJ']
                # Additional records
                if aj_flags['KBK'] > 0:
                    lbk_list_keys = ((),(),#('ED','EU',0,0,'LBK',0),
                                     ('R0','R1','R2','S0','S1',0),
                                     ('R0','SO','GA',0,0,0))
                    total_lines = read_kbks(nch, subsection, aj_data, total_lines)
                if aj_flags['KPS'] > 0:
                    total_lines = read_kpss(nch, subsection, aj_data, total_lines)
                subsection_dict[aj] = aj_data

        el, eh = range_flags['EL'], range_flags['EH']
        subsection_data = (el,eh,subsection_dict,range_flags)
        isotope_dict = self.structure[mat_id]['data'][nuc_i]
        isotope_dict['resolved'].append(subsection_data)
        return total_lines

    def _read_unresolved(self, subsection, range_flags, isotope_flags, mat_id,
                         nuc_i):
        """Read unresolved resonances of an energy subsection.

        Parameters
        -----------
        subsection: array
            Contains data for energy subsection.
        range_flags: dict
            Contains metadata flags for energy range.
        isotope_flags: dict
            Contiains flags for isotope.
        mat_id: int
            Material ZZAAAM.
        nuc_i: int
            Isotope ZZAAAM.

        Returns
        --------
        total_lines: int
        """
        head_cont = ('SPI','AP','LSSF',0,'NLS',0)
        has_head_cont = {(0,1): True, (1,1): False, (0,2): True, (1,2): True}
        L_keys = {(0,1): ('AWRI',0,'L',0,'6*NJS','NJS'),
                  (1,1): ('AWRI',0,'L',0,'NJS',0),
                  (0,2): ('AWRI',0,'L',0,'NJS',0),
                  (1,2): ('AWRI',0,'L',0,'NJS',0)}
        j_keys = {(0,1): None,
                  (1,1): (0,0,'L','MUF','NE+6',0,'D','AJ','AMUN','GN0','GG',
                          0),
                  (0,2): ('AJ',0,'INT',0,'6*NE+6','NE',0,0,'AMUX','AMUN',
                      'AMUG','AMUF'),
                  (1,2): ('AJ',0,'INT',0,'6*NE+6','NE',0,0,'AMUX','AMUN',
                      'AMUG','AMUF')}
        itemkeys = {(0,1): ('D','AJ','AMUN','GN0','GG',0),
                    (1,1): ('GF',),
                    (0,2): ('ES','D','GX','GN0','GG','GF'),
                    (1,2): ('ES','D','GX','GN0','GG','GF')}

        lfw, lrf = int(isotope_flags['LFW']), int(range_flags['LRF'])
        subsection_dict = rx.DoubleSpinDict({})
        if range_flags['NRO'] > 0:
            tabhead,intdata,total_lines=self._get_tab1((0,0,0,0,'NR','NP'),
                                                       ('E','AP'),
                                                       subsection)
            subsection_dict['int']= intdata
        else:
            total_lines = 0
        if has_head_cont[(lfw, lrf)]:
            range_flags, total_lines = self._cont_and_update(
                range_flags, head_cont, subsection, total_lines)
        if (lfw, lrf) == (1,1):
            # Case B in ENDF manual p.70
            head_flags, es_array, lines = self._get_list(
                ('SPI','AP','LSSF',0,'NE','NLS'),
                ('ES',),
                subsection[total_lines:])
            subsection_dict['ES'] = es_array['ES']
            total_lines += lines
            range_flags.update(head_flags)
        total_lines = self._nls_njs_loop(L_keys[(lfw, lrf)],
                                        j_keys[(lfw, lrf)],
                                        itemkeys[(lfw, lrf)],
                                        subsection,
                                        total_lines,
                                        range_flags,
                                        subsection_dict)
        el, eh = range_flags['EL'], range_flags['EH']
        subsection_data = (el,eh,subsection_dict,range_flags)
        isotope_dict = self.structure[mat_id]['data'][nuc_i]
        isotope_dict['unresolved'].append(subsection_data)
        return total_lines

    def _read_ap_only(self, subsection, range_flags, isotope_flags, mat_id,
                      nuc_i):
        "Read in scattering radius when it is the only resonance data given."
        subsection_dict = {}
        if range_flags['NRO'] > 0:
            tabhead,intdata,total_lines=self._get_tab1((0,0,0,0,'NR','NP'),
                                                       ('E','AP'),
                                                       subsection)
            subsection_dict['int']= intdata
        else:
            total_lines = 0
        range_flags, total_lines = self._cont_and_update(
            range_flags, ('SPI','AP',0,0,'NLS',0), subsection, total_lines)
        return total_lines

    def _read_xs(self, nuc, mt, nuc_i=None):
        """Read in cross-section data. Read resonances with Library._read_res
        first.

        Parameters
        -----------
        nuc: int
            id of material.
        mt: int
            Reaction number to find cross-section data of.
        nuc_i: int
            Isotope to find; if None, defaults to mat_id.
        """
        nuc = nucname.id(nuc)
        if nuc_i == None:
            nuc_i = nuc
        xsdata = self.get_rx(nuc, 3, mt).reshape(-1,6)
        total_lines = 0
        head_flags = self._get_head(('ZA','AWR',0,0,0,0),
                                    xsdata[total_lines])
        total_lines += 1
        int_flags, int_data, int_size = self._get_tab1(
            ('QM','QI',0,'LM','NR','NP'),
            ('e_int','xs'),
            xsdata[total_lines:])
        int_flags.update(head_flags)
        isotope_dict = self.structure[nuc]['data'][nuc_i]
        isotope_dict['xs'].update({mt: (int_data, int_flags)})
        total_lines += int_size

    def get_xs(self, nuc, mt, nuc_i=None):
        """get_xs(nuc, mt, nuc_i=None)
        Grab cross-section data.

        Parameters
        -----------
        nuc: int
            id of nuclide to read.
        mt: int
            ENDF reaction number to read.
        nuc_i: int
            id of isotope to read. Defaults to nuc.

        Returns
        --------
        tuple
            Returns a tuple with xs data in tuple[0] and flags in tuple[1].
        """
        nuc = nucname.id(nuc)
        if not nuc_i:
            nuc_i = nuc
        else:
            nuc_i = nucname.id(nuc_i)
        if (nuc not in self.structure) or (not self.structure[nuc]['data']):
            self._read_res(nuc)
        if nuc_i not in self.structure[nuc]['data'] or \
           mt not in self.structure[nuc]['data'][nuc_i]['xs']:
            self._read_xs(nuc, mt, nuc_i)
        return self.structure[nuc]['data'][nuc_i]['xs'][mt]

    def get_rx(self, nuc, mf, mt, lines=0):
        """get_rx(nuc, mf, mt, lines=0)
        Grab the data from one reaction type.

        Parameters
        -----------
        nuc: int
            id form of material to read from.
        mf: int
            ENDF file number (MF).
        mt: int
            ENDF reaction number (MT).
        lines: int
            Number of lines to read from this reaction, starting from the top.
            Default value is 0, which reads in the entire reaction.

        Returns
        --------
        data: NumPy array
            Contains the reaction data in an Nx6 array.
        """
        nuc = nucname.id(nuc)
        if nuc in self.structure:
            return self._read_nucmfmt(nuc, mf, mt, lines)
        else:
            raise ValueError("Material {} does not exist.".format(nuc))

    def _read_nucmfmt(self, nuc, mf, mt, lines):
        """Load in the data from one reaction into self.structure.

        Parameters
        -----------
        nuc : int
            id of nuclide.
        mf : int
            ENDF file number (MF).
        mt : int
            ENDF reaction number (MT).

        Returns
        --------
        array, 1d, float64
            1d, float64 NumPy array containing the reaction data.
        """
        opened_here = False
        if isinstance(self.fh, basestring):
            fh = open(self.fh, 'r')
            opened_here = True
        else:
            fh = self.fh
        try:
            start, stop = self.mat_dict[nuc]['mfs'][mf,mt]
        except KeyError as e:
            msg = "MT {1} not found in File {0}.".format(mf, mt)
            e.args = (msg,)
            raise e
        fh.readline()
        fh.seek(start)
        if lines == 0:
            s = fh.read(stop-start)
        else:
            s = fh.read(lines*81)
        if opened_here:
            fh.close
        return fromendf_tok(s)

class Evaluation(object):
    """
    Evaluation is the main class for an ENDF evaluation which contains a number
    of Files.
    """

    def __init__(self, filename_or_handle, verbose=True):
        if isinstance(filename_or_handle, file):
            self._fh = filename_or_handle
        else:
            self._fh = open(filename_or_handle, 'r')
        self.files = []
        self._verbose = verbose
        self._veryverbose = False

        # Determine MAT number for this evaluation
        MF = 0
        while MF == 0:
            position = self._fh.tell()
            line = self._fh.readline()
            MF = int(line[70:72])
        self.material = int(line[66:70])

        # Save starting position for this evaluation
        self._start_position = position
        self._fh.seek(position)

        # Create list for reactions
        self.reactions = OrderedDict()

        # First we need to read MT=1, MT=451 which has a description of the ENDF
        # file and a list of what data exists in the file
        self._read_header()

    def read(self, reactions=None):
        if not reactions:
            if self._verbose:
                print("No reaction given. Read all")
            reactions = []
            for r in self.reaction_list[1:]:
                reactions.append(r[0:2])
        if isinstance(reactions, tuple):
            reactions = [reactions]
        # Start looping over the requested reactions entry since it is the
        # MT=451 block that we already read
        for rMF, rMT in reactions:
            found = False
            for MF, MT, NC, MOD in self.reaction_list:
                if MF == rMF and MT == rMT:
                    found = True
                    # File 1 data
                    if MF == 1:
                        # Number of total neutrons per fission
                        if MT == 452:
                            self._read_total_nu()
                        # Number of delayed neutrons per fission
                        elif MT == 455:
                            self._read_delayed_nu()
                        # Number of prompt neutrons per fission
                        elif MT == 456:
                            self._read_prompt_nu()
                        # Components of energy release due to fission
                        elif MT == 458:
                            self._read_fission_energy()
                        elif MT == 460:
                            self._read_delayed_photon()

                    elif MF == 2:
                        # File 2 -- Resonance parameters
                        if MT == 151:
                            self._read_resonances()

                    elif MF == 3:
                        # File 3 -- Reaction cross sections
                        self._read_reaction_xs(MT)

                    elif MF == 5:
                        # File 5 -- Energy distributions
                        self._read_energy_distribution(MT)

                    elif MF == 7:
                        # File 7 -- Thermal scattering data
                        if MT == 2:
                            self._read_thermal_elastic()
                        if MT == 4:
                            self._read_thermal_inelastic()

                    elif MF == 8:
                        # Read File 8 -- decay and fission yield data
                        if MT == 454:
                            self._read_independent_yield()
                        elif MT == 459:
                            self._read_cumulative_yield()
                        elif MT == 457:
                            self._read_decay()

                    elif MF == 9:
                        # Read File 9 -- multiplicities
                        self._read_multiplicity(MT)

                    elif MF == 10:
                        # Read File 10 -- cross sections for production of
                        # radioactive nuclides
                        self._read_production_xs(MT)

            if not found:
                if self._verbose:
                    print("Reaction not found")
                raise NotFound('Reaction')

    def _read_header(self):
        self._print_info(1, 451)
        self._seek_mfmt(1, 451)

        # Information about target/projectile
        self.target = {}
        self.projectile = {}
        self.info = {}

        # First HEAD record
        items = self._get_head_record()
        self.target['ZA'] = items[0]
        self.target['mass'] = items[1]
        self._LRP = items[2]
        self.target['fissionable'] = (items[3] == 1)
        try:
            global libraries
            library = libraries[items[4]]
        except KeyError:
            library = 'Unknown'
        self.info['modification'] = items[5]

        # Create dictionary for nu if fissionable
        if self.target['fissionable']:
            self.nu = {}

        # Control record 1
        items = self._get_cont_record()
        self.target['excitation_energy'] = items[0]
        self.target['stable'] = (int(items[1]) == 0)
        self.target['state'] = items[2]
        self.target['isomeric_state'] = items[3]
        self.format = items[5]
        assert self.format == 6

        # Control record 2
        items = self._get_cont_record()
        self.projectile['mass'] = items[0]
        self.energy_max = items[1]
        library_release = items[2]
        self.sublibrary = items[4]
        library_version = items[5]
        self.info['library'] = (library, library_version, library_release)

        # Control record 3
        items = self._get_cont_record()
        self.target['temperature'] = items[0]
        self.info['derived'] = (items[2] > 0)
        NWD = items[4]
        NXC = items[5]

        # Text record 1
        items = self._get_text_record()
        text = items[0]
        self.target['zsymam'] = text[0:11]
        self.info['laboratory'] = text[11:22]
        self.info['date'] = text[22:32]
        self.info['author'] = text[32:66]

        # Text record 2
        items = self._get_text_record()
        text = items[0]
        self.info['reference'] = text[1:22]
        self.info['date_distribution'] = text[22:32]
        self.info['date_release'] = text[33:43]
        self.info['date_entry'] = text[55:63]

        # Text records 3-5
        items0 = self._get_text_record()
        items1 = self._get_text_record()
        items2 = self._get_text_record()
        self.info['identifier'] = [items0[0], items1[0], items2[0]]

        # Now read descriptive records
        self.info['description'] = []
        for i in range(NWD - 5):
            line = self._fh.readline()[:66]
            self.info['description'].append(line)

        # File numbers, reaction designations, and number of records
        self.reaction_list = []
        for i in range(NXC):
            items = self._get_cont_record(skipC=True)
            MF, MT, NC, MOD = items[2:6]
            self.reaction_list.append((MF,MT,NC,MOD))

    def _read_total_nu(self):
        self._print_info(1, 452)
        self._seek_mfmt(1, 452)

        # Create total nu reaction
        self.nu['total'] = {}

        # Determine representation of total nu data
        items = self._get_head_record()
        LNU = items[3]

        # Polynomial representation
        if LNU == 1:
            self.nu['total']['form'] = 'polynomial'
            self.nu['total']['coefficients'] = np.asarray(
                self._get_list_record(onlyList=True))
        # Tabulated representation
        elif LNU == 2:
            self.nu['total']['form'] = 'tabulated'
            params, self.nu['total']['values'] = self._get_tab1_record()

        # Skip SEND record
        self._fh.readline()

    def _read_delayed_nu(self):
        self._print_info(1, 455)
        self._seek_mfmt(1, 455)

        # Create delayed nu reaction
        self.nu['delayed'] = {}

        # Determine representation of delayed nu data
        items = self._get_head_record()
        LDG = items[2]
        LNU = items[3]
        self.nu['delayed']['decay_energy_dependent'] = (LDG == 1)

        if LDG == 0:
            # Delayed-group constants energy independent
            self.nu['delayed']['decay_constants'] = np.asarray(
                self._get_list_record(onlyList=True))
        elif LDG == 1:
            # Delayed-group constants energy dependent
            raise NotImplementedError

        if LNU == 1:
            # Nu represented as polynomial
            self.nu['delayed']['form'] = 'polynomial'
            self.nu['delayed']['coefficients'] = np.asarray(
                self._get_list_record(onlyList=True))
        elif LNU == 2:
            self.nu['delayed']['form'] = 'tabulated'
            params, self.nu['delayed']['values'] = self._get_tab1_record()
        self._fh.readline()

    def _read_prompt_nu(self):
        self._print_info(1, 456)
        self._seek_mfmt(1, 456)

        # Create delayed nu reaction
        self.nu['prompt'] = {}

        # Determine representation of delayed nu data
        items = self._get_head_record()
        LNU = items[3]

        if LNU == 1:
            # Polynomial representation (spontaneous fission)
            self.nu['prompt']['form'] = 'polynomial'
            self.nu['prompt']['coefficients'] = np.asarray(
                self._get_list_record(onlyList=True))
        elif LNU == 2:
            # Tabulated values of nu
            self.nu['prompt']['form'] = 'tabulated'
            params, self.nu['prompt']['values'] = self._get_tab1_record()

        # Skip SEND record
        self._fh.readline()

    def _read_fission_energy(self):
        self._print_info(1, 458)
        self._seek_mfmt(1, 458)

        # Create fission energy release reaction
        er = {}
        self.fission_energy_release = er

        # Skip HEAD record
        self._get_head_record()

        # Read LIST record containing components of fission energy release (or
        # coefficients)
        items, values = self._get_list_record()
        NPLY = items[3]
        er['order'] = NPLY

        values = np.asarray(values)
        values.shape = (NPLY + 1, 18)
        er['fission_products'] = np.vstack(values[:,0], values[:,1])
        er['prompt_neutrons'] = np.vstack(values[:,2], values[:,3])
        er['delayed_neutrons'] = np.vstack(values[:,4], values[:,5])
        er['prompt_gammas'] = np.vstack(values[:,6], values[:,7])
        er['delayed_gammas'] = np.vstack(values[:,8], values[:,9])
        er['delayed_betas'] = np.vstack(values[:,10], values[:,11])
        er['neutrinos'] = np.vstack(values[:,12], values[:,13])
        er['total_less_neutrinos'] = np.vstack(values[:,14], values[:,15])
        er['total'] = np.vstack(values[:,16], values[:,17])

        # Skip SEND record
        self._fh.readline()

    def _read_reaction_xs(self, MT):
        self._print_info(3, MT)
        self._seek_mfmt(3, MT)

        # Get Reaction instance
        if MT not in self.reactions:
            self.reactions[MT] = Reaction(MT)
        rxn = self.reactions[MT]
        rxn.MFs.append(3)

        # Read HEAD record with ZA and atomic mass ratio
        items = self._get_head_record()

        # Read TAB1 record with reaction cross section
        params, rxn.cross_section = self._get_tab1_record()
        rxn.Q_mass_difference = params[0]
        rxn.Q_reaction = params[1]
        rxn.complex_breakup_flag = params[3]

        # Skip SEND record
        self._fh.readline()

    def _read_energy_distribution(self, MT):
        # Find energy distribution
        self._print_info(5, MT)
        self._seek_mfmt(5, MT)

        # Get Reaction instance
        if MT not in self.reactions:
            self.reactions[MT] = Reaction(MT)
        rxn = self.reactions[MT]
        rxn.MFs.append(5)

        # Read HEAD record
        items = self._get_head_record()
        nk = items[4]

        rxn.distributions = []
        for i in range(nk):
            edist = EnergyDistribution()

            # Read TAB1 record for p(E)
            params, applicability = self._get_tab1_record()
            lf = params[3]
            if lf == 1:
                edist.tab2 = self._get_tab2_record()
                n_energies = edist.tab2.params[5]

                edist.func_list = []
                edist.energy_in = np.zeros(n_energies)
                for j in range(n_energies):
                    params, func = self._get_tab1_record()
                    self.energy_in[j] = params[1]
                    edist.func_list.append(func)
            elif lf == 5:
                # General evaporation spectrum
                edist.u = params[0]
                params, edist.theta = self._get_tab1_record()
                params, edist.g = self._get_tab1_record()
            elif lf == 7:
                # Simple Maxwellian fission spectrum
                edist.u = params[0]
                params, edist.theta = self._get_tab1_record()
            elif lf == 9:
                # Evaporation spectrum
                edist.u = params[0]
                params, edist.theta = self._get_tab1_record()
            elif lf == 11:
                # Energy-dependent Watt spectrum
                edist.u = params[0]
                params, edist.a = self._get_tab1_record()
                params, edist.b = self._get_tab1_record()
            elif lf == 12:
                # Energy-dependent fission neutron spectrum (Madland-Nix)
                params, edist.tm = self._get_tab1_record()
                edist.efl, edist.efh = params[0:2]

            edist.lf = lf
            edist.applicability = applicability
            rxn.distributions.append(edist)

    def _read_delayed_photon(self):
        self._print_info(1, 460)
        self._seek_mfmt(1, 460)

        # Create delayed photon data reaction
        dp = {}
        self.delayed_photons = dp

        # Determine whether discrete or continuous representation
        items = self._get_head_record()
        LO = items[2]
        NG = items[4]

        # Discrete representation
        if LO == 1:
            dp['form'] = 'discrete'

            # Initialize lists for energies of photons and time dependence of
            # photon multiplicity
            dp['energy'] = np.zeros(NG)
            dp['multiplicity'] = []
            for i in range(NG):
                # Read TAB1 record with multiplicity as function of time
                params, mult = self._get_tab1_record()
                dp['multiplicity'].append(mult)

                # Determine energy
                dp['energy'][i] = params[0]

        # Continuous representation
        elif LO == 2:
            # Determine decay constant and number of precursor families
            dp['form'] = 'continuous'
            dp['decay_constant'] = self._get_list_record(onlyList=True)

    def _read_resonances(self):
        self._print_info(2, 151)
        self._seek_mfmt(2, 151)

        # Create MT for resonances
        res = {}
        self.resonances = res

        # Determine whether discrete or continuous representation
        items = self._get_head_record()
        NIS = items[4] # Number of isotopes
        res['isotopes'] = []

        for iso in range(NIS):
            # Create dictionary for this isotope
            isotope = {}
            res['isotopes'].append(isotope)

            items = self._get_cont_record()
            isotope['abundance'] = items[1]
            LFW = items[3] # average fission width flag
            NER = items[4] # number of resonance energy ranges

            isotope['ranges'] = []

            for j in range(NER):
                # Create dictionary for energy range
                erange = {}
                isotope['ranges'].append(erange)

                items = self._get_cont_record()
                erange['energy_min'] = items[0]
                erange['energy_max'] = items[1]
                LRU = items[2]  # flag for resolved (1)/unresolved (2)
                LRF = items[3]  # resonance representation
                erange['NRO'] = items[4]  # flag for energy dependence of scattering radius
                erange['NAPS'] = items[5]  # flag controlling use of channel/scattering radius

                if LRU == 0 and erange['NRO'] == 0:
                    # Only scattering radius specified
                    erange['type'] = 'scattering_radius'
                    items = self._get_cont_record()
                    erange['spin'] = items[0]
                    erange['scattering_radius'] = items[1]
                elif LRU == 1:
                    # resolved resonance region
                    erange['type'] = 'resolved'
                    erange['representation'] = resonance_types[LRF]
                    self._read_resolved(erange)
                    if NIS == 1:
                        res['resolved'] = erange
                elif LRU == 2:
                    # unresolved resonance region
                    erange['type'] = 'unresolved'
                    self._read_unresolved(erange, LFW, LRF)
                    if NIS == 1:
                        res['unresolved'] = erange

    def _read_resolved(self, erange):
        if erange['representation'] in ('SLBW', 'MLBW'):
            # -------------- Single- or Multi-level Breit Wigner ---------------

            # Read energy-dependent scattering radius if present
            if erange['NRO'] != 0:
                params, erange['scattering_radius'] = self._get_tab1_record()

            # Other scatter radius parameters
            items = self._get_cont_record()
            erange['spin'] = items[0]
            if erange['NRO'] == 0:
                erange['scattering_radius'] = items[1]
            NLS = items[4]  # Number of l-values

            erange['resonances'] = []

            # Read resonance widths, J values, etc
            for l in range(NLS):
                headerItems, items = self._get_list_record()
                QX, L, LRX = headerItems[1:4]
                energy = items[0::6]
                spin = items[1::6]
                GT = items[2::6]
                GN = items[3::6]
                GG = items[4::6]
                GF = items[5::6]
                for i, E in enumerate(energy):
                    resonance = BreitWigner()
                    resonance.QX = QX
                    resonance.L = L
                    resonance.LRX = LRX
                    resonance.E = energy[i]
                    resonance.J = spin[i]
                    resonance.GT = GT[i]
                    resonance.GN = GN[i]
                    resonance.GG = GG[i]
                    resonance.GF = GF[i]
                    erange['resonances'].append(resonance)

        elif erange['representation'] == 'RM':
            # ------------------------- Reich-Moore ----------------------------

            # Read energy-dependent scattering radius if present
            if erange['NRO'] != 0:
                params, erange['scattering_radius'] = self._get_tab1_record()

            # Other scatter radius parameters
            items = self._get_cont_record()
            erange['spin'] = items[0]
            if erange['NRO'] == 0:
                erange['scattering_radius'] = items[1]
            erange['LAD'] = items[3]  # Flag for angular distribution
            NLS = items[4]  # Number of l-values
            NLSC = items[5]  # Number of l-values for convergence

            erange['resonances'] = []

            # Read resonance widths, J values, etc
            for l in range(NLS):
                headerItems, items = self._get_list_record()
                APL, L = headerItems[1:3]
                energy = items[0::6]
                spin = items[1::6]
                GN = items[2::6]
                GG = items[3::6]
                GFA = items[4::6]
                GFB = items[5::6]
                for i, E in enumerate(energy):
                    resonance = ReichMoore()
                    resonance.APL = APL
                    resonance.L = L
                    resonance.E = energy[i]
                    resonance.J = spin[i]
                    resonance.GN = GN[i]
                    resonance.GG = GG[i]
                    resonance.GFA = GFA[i]
                    resonance.GFB = GFB[i]
                    erange['resonances'].append(resonance)

        elif erange['representation'] == 'AA':
            # --------------------------- Adler-Adler --------------------------

            # Read energy-dependent scattering radius if present
            if erange['NRO'] != 0:
                params, erange['scattering_radius'] = self._get_tab1_record()

            # Other scatter radius parameters
            items = self._get_cont_record()
            erange['spin'] = items[0]
            if erange['NRO'] == 0:
                erange['scattering_radius'] = items[1]
            NLS = items[4]  # Number of l-values

            # Get AT, BT, AF, BF, AC, BC constants
            items, values = self._get_list_record()
            erange['LI'] = items[2]
            NX = items[5]
            erange['AT'] = np.asarray(values[:4])
            erange['BT'] = np.asarray(values[4:6])
            if NX == 2:
                erange['AC'] = np.asarray(values[6:10])
                erange['BC'] = np.asarray(values[10:12])
            elif NX == 3:
                erange['AF'] = np.asarray(values[6:10])
                erange['BF'] = np.asarray(values[10:12])
                erange['AC'] = np.asarray(values[12:16])
                erange['BC'] = np.asarray(values[16:18])

            erange['resonances'] = []

            for ls in range(NLS):
                items = self._get_cont_record()
                l_value = items[2]
                NJS = items[4]
                for j in range(NJS):
                    items, values = self._get_list_record()
                    AJ = items[0]
                    NLJ = items[5]
                    for res in range(NLJ):
                        resonance = AdlerAdler()
                        resonance.L, resonance.J = l_value, AJ
                        resonance.DET = values[12*res]
                        resonance.DWT = values[12*res + 1]
                        resonance.DRT = values[12*res + 2]
                        resonance.DIT = values[12*res + 3]
                        resonance.DEF_ = values[12*res + 4]
                        resonance.DWF = values[12*res + 5]
                        resonance.GRG = values[12*res + 6]
                        resonance.GIF = values[12*res + 7]
                        resonance.DEC = values[12*res + 8]
                        resonance.DWC = values[12*res + 9]
                        resonance.DRC = values[12*res + 10]
                        resonance.DIC = values[12*res + 11]
                        erange['resonances'].append(resonance)

        elif erange['representation'] == 'RML':
            pass

    def _read_unresolved(self, erange, LFW, LRF):
        erange['fission_widths'] = (LFW == 1)
        erange['LRF'] = LRF

        # Read energy-dependent scattering radius if present
        if erange['NRO'] != 0:
            params, erange['scattering_radius'] = self._get_tab1_record()

        # Get SPI, AP, and LSSF
        if not (LFW == 1 and LRF == 1):
            items = self._get_cont_record()
            erange['spin'] = items[0]
            if erange['NRO'] == 0:
                erange['scatter_radius'] = items[1]
            erange['LSSF'] = items[2]

        if LFW == 0 and LRF == 1:
            # Case A -- fission widths not given, all parameters are
            # energy-independent
            NLS = items[4]
            erange['l_values'] = np.zeros(NLS)
            erange['parameters'] = {}
            for ls in range(NLS):
                items, values = self._get_list_record()
                l = items[2]
                NJS = items[5]
                erange['l_values'][ls] = l
                params = {}
                erange['parameters'][l] = params
                params['d'] = np.asarray(values[0::6])
                params['j'] = np.asarray(values[1::6])
                params['amun'] = np.asarray(values[2::6])
                params['gn0'] = np.asarray(values[3::6])
                params['gg'] = np.asarray(values[4::6])
                # params['gf'] = np.zeros(NJS)

        elif LFW == 1 and LRF == 1:
            # Case B -- fission widths given, only fission widths are
            # energy-dependent
            items, erange['energies'] = self._get_list_record()
            erange['spin'] = items[0]
            if erange['NRO'] == 0:
                erange['scatter_radius'] = items[1]
            erange['LSSF'] = items[2]
            NE, NLS = items[4:6]
            erange['l_values'] = np.zeros(NLS)
            erange['parameters'] = {}
            for ls in range(NLS):
                items = self._get_cont_record()
                l = items[2]
                NJS = items[4]
                erange['l_values'][ls] = l
                params = {}
                erange['parameters'][l] = params
                params['d'] = np.zeros(NJS)
                params['j'] = np.zeros(NJS)
                params['amun'] = np.zeros(NJS)
                params['gn0'] = np.zeros(NJS)
                params['gg'] = np.zeros(NJS)
                params['gf'] = []
                for j in range(NJS):
                    items, values = self._get_list_record()
                    muf = items[3]
                    params['d'][j] = values[0]
                    params['j'][j] = values[1]
                    params['amun'][j] = values[2]
                    params['gn0'][j] = values[3]
                    params['gg'][j] = values[4]
                    params['gf'].append(np.asarray(values[6:]))

        elif LRF == 2:
            # Case C -- all parameters are energy-dependent
            NLS = items[4]
            erange['l_values'] = np.zeros(NLS)
            erange['parameters'] = {}
            for ls in range(NLS):
                items = self._get_cont_record()
                l = items[2]
                NJS = items[4]
                erange['l_values'][ls] = l
                params = {}
                erange['parameters'][l] = params
                params['j'] = np.zeros(NJS)
                params['amux'] = np.zeros(NJS)
                params['amun'] = np.zeros(NJS)
                params['amug'] = np.zeros(NJS)
                params['amuf'] = np.zeros(NJS)
                params['energies'] = []
                params['d'] = []
                params['gx'] = []
                params['gn0'] = []
                params['gg'] = []
                params['gf'] = []
                for j in range(NJS):
                    items, values = self._get_list_record()
                    ne = items[5]
                    params['j'][j] = items[0]
                    params['amux'][j] = values[2]
                    params['amun'][j] = values[3]
                    params['amug'][j] = values[4]
                    params['amuf'][j] = values[5]
                    params['energies'].append(np.asarray(values[6::6]))
                    params['d'].append(np.asarray(values[7::6]))
                    params['gx'].append(np.asarray(values[8::6]))
                    params['gn0'].append(np.asarray(values[9::6]))
                    params['gg'].append(np.asarray(values[10::6]))
                    params['gf'].append(np.asarray(values[11::6]))

    def _read_thermal_elastic(self):
        self._print_info(7, 2)
        self._seek_mfmt(7, 2)

        # Create dictionary for thermal elastic data
        elast = {}
        self.thermal_elastic = elast

        # Get head record
        items = self._get_head_record()
        LTHR = items[2]  # coherent/incoherent flag
        elast['S'] = {}

        if LTHR == 1:
            elast['type'] = 'coherent'
            params, sdata = self._get_tab1_record()
            temperature = params[0]
            LT = params[2]
            elast['S'][temperature] = sdata

            for t in range(LT):
                params, sdata = self._get_list_record()
                temperature = params[0]
                LT = params[2]
                elast['S'][temperature] = sdata

        elif elast.LTHR == 2:
            elast['type'] = 'incoherent'
            params, wt = self._get_tab1_record()
            elast['bound_cross_section'] = params[0]
            elast['debye_waller'] = wt

    def _read_thermal_inelastic(self):
        self._print_info(7, 4)
        self._seek_mfmt(7, 4)

        # Create dictionary for thermal inelstic data
        inel = {}
        self.thermal_inelastic = inel

        # Get head record
        items = self._get_head_record()
        inel['LAT'] = items[3]  # Temperature flag
        inel['LASYM'] = items[4]  # Symmetry flag
        header, B = self._get_list_record()
        inel['LLN'] = header[2]
        inel['NS'] = header[5]
        inel['B'] = B
        if B[0] != 0.0:
            nbeta = self._get_tab2_record()
            sabt = []
            beta = []
            for be in range(nbeta.NBT[0]):
                #Read record for first temperature (always present)
                sabt_temp = []
                temp = []
                params, temp0 = self._get_tab1_record()
                # Save S(be, 0, :)
                sabt_temp.append(temp0)
                beta.append(params[1])
                temp.append(params[0])
                LT = temp0.params[2]
                for t in range(LT):
                    # Read records for all the other temperatures
                    headsab, sa = self._get_list_record()
                    sabt_temp.append(sa)
                    temp.append(headsab[0])
                sabt.append(sabt_temp)

            # Prepare arrays for output
            inel['sabt'] = sabt
            inel['beta'] = np.array(beta)
            inel['temp'] = np.array(temp)

        params, teff = self._get_tab1_record()
        inel['teff'] = teff

    def _read_independent_yield(self):
        self._print_info(8, 454)
        self._seek_mfmt(8, 454)

        # Create dictionary for independent yield
        iyield = {}
        self.yield_independent = iyield

        # Initialize energies and yield dictionary
        iyield['energies'] = []
        iyield['data'] = {}
        iyield['interp'] = []

        items = self._get_head_record()
        iyield['ZA'] = items[0]
        iyield['AWR'] = items[1]
        LE = items[2] - 1  # Determine energy-dependence

        for i in range(LE):
            items, itemList = self._get_list_record()
            E = items[0]  # Incident particle energy
            iyield['energies'].append(E)
            NFP = items[5]  # Number of fission product nuclide states
            if i > 0:
                iyield['interp'].append(items[2]) # Interpolation scheme

            # Get data for each yield
            iyield['data'][E] = {}
            iyield['data'][E]['zafp'] = [int(i) for i in itemList[0::4]] # ZA for fission products
            iyield['data'][E]['fps'] = itemList[1::4] # State designator
            iyield['data'][E]['yi'] = zip(itemList[2::4],itemList[3::4]) # Independent yield

        # Skip SEND record
        self._fh.readline()

    def _read_cumulative_yield(self):
        self._print_info(8, 459)
        self._seek_mfmt(8, 459)

        # Create dictionary for cumulative yield
        cyield = {}
        self.yield_cumulative = cyield

        # Initialize energies and yield dictionary
        cyield['energies'] = []
        cyield['data'] = {}
        cyield['interp'] = []

        items = self._get_head_record()
        cyield['ZA'] = items[0]
        cyield['AWR'] = items[1]
        LE = items[2] - 1  # Determine energy-dependence

        for i in range(LE):
            items, itemList = self._get_list_record()
            E = items[0]  # Incident particle energy
            cyield['energies'].append(E)
            NFP = items[5]  # Number of fission product nuclide states
            if i > 0:
                cyield['interp'].append(items[2]) # Interpolation scheme

            # Get data for each yield
            cyield['data'][E] = {}
            cyield['data'][E]['zafp'] = [int(i) for i in itemList[0::4]] # ZA for fission products
            cyield['data'][E]['fps'] = itemList[1::4] # State designator
            cyield['data'][E]['yc'] = zip(itemList[2::4],itemList[3::4]) # Cumulative yield

        # Skip SEND record
        self._fh.readline()

    def _read_decay(self):
        self._print_info(8, 457)
        self._seek_mfmt(8, 457)

        decay = {}
        self.decay = decay

        # Get head record
        items = self._get_head_record()
        decay['ZA'] = items[0]  # ZA identifier
        decay['awr'] = items[1]  # AWR
        decay['state']= items[2]  # State of the original nuclide
        decay['isomeric_state'] = items[3]  # Isomeric state for the original nuclide
        decay['stable'] = (items[4] == 1)  # Nucleus stability flag

        # Determine if radioactive/stable
        if not decay['stable']:
            NSP = items[5]  # Number of radiation types

            # Half-life and decay energies
            items, itemList = self._get_list_record()
            decay['half_life'] = (items[0], items[1])
            decay['NC'] = items[4]//2
            decay['energies'] = zip(itemList[0::2], itemList[1::2])

            # Decay mode information
            items, itemList = self._get_list_record()
            decay['spin'] = items[0]  # Spin of the nuclide
            decay['parity'] = items[1]  # Parity of the nuclide
            NDK = items[5]  # Number of decay modes

            # Decay type (beta, gamma, etc.)
            decay['modes'] = {}
            decay['modes']['type'] = []
            for i in itemList[0::6]:
                if i % 1.0 == 0:
                    decay['modes']['type'].append(radiation_type[int(i)])
                else:
                    decay['modes']['type'].append(
                        (radiation_type[int(i)],
                         radiation_type[int(10*i % 10)]))
            decay['modes']['isomeric_state'] = itemList[1::6]
            decay['modes']['energy'] = zip(itemList[2::6], itemList[3::6])
            decay['modes']['branching_ratio'] = zip(itemList[4::6], itemList[5::6])

            # Read spectra
            decay['spectra'] = {}
            for i in range(NSP):
                items, itemList = self._get_list_record()
                STYP = radiation_type[items[1]]  # Decay radiation type
                LCON = items[2]  # Continuous spectrum flag
                NER = items[5]  # Number of tabulated discrete energies

                if LCON != 1:
                    for j in range(NER):
                        items, itemList = self._get_list_record()

                if LCON != 0:
                    r = self._get_tab1_record()
                    LCOV = r.params[3]

                    if LCOV != 0:
                        items, itemList = self._get_list_record()

                decay['spectra'][STYP] = {'LCON': LCON, 'NER': NER}

        else:
            items, itemList = self._get_list_record()
            items, itemList = self._get_list_record()
            decay['spin'] = items[0]
            decay['parity'] = items[1]

        # Skip SEND record
        self._fh.readline()

    def _read_multiplicity(self, MT):
        self._print_info(9, MT)
        self._seek_mfmt(9, MT)

        # Get Reaction instance
        if MT not in self.reactions:
            self.reactions[MT] = Reaction(MT)
        rxn = self.reactions[MT]
        rxn.MFs.append(9)

        # Get head record
        items = self._get_head_record()
        NS = items[4]  # Number of final states

        rxn.multiplicities = {}
        for i in range(NS):
            params, state = self._get_tab1_record()
            QM = params[0] # Mass difference Q value (eV)
            QI = params[1] # Reaction Q value (eV)
            IZAP = params[2] # 1000Z + A
            LFS = params[3] # Level number of the nuclide
            rxn.multiplicities[LFS] = {'QM': QM, 'QI': QI, 'ZA': IZAP,
                                       'values': state}

    def _read_production_xs(self, MT):
        self._print_info(10, MT)
        self._seek_mfmt(10, MT)

        # Get Reaction instance
        if MT not in self.reactions:
            self.reactions[MT] = Reaction(MT)
        rxn = self.reactions[MT]
        rxn.MFs.append(10)

        # Get head record
        items = self._get_head_record()
        NS = items[4]  # Number of final states

        rxn.production = {}
        for i in range(NS):
            params, state = self._get_tab1_record()
            QM = params[0]  # Mass difference Q value (eV)
            QI = params[1]  # Reaction Q value (eV)
            IZAP = params[2]  # 1000Z + A
            LFS = params[3]  # Level number of the nuclide
            rxn.production[LFS] = {'QM': QM, 'QI': QI, 'ZA': IZAP,
                                   'values': state}

    def _get_text_record(self, line=None):
        if not line:
            line = self._fh.readline()
        if self._veryverbose:
            print("Get TEXT record")
        HL = line[0:66]
        MAT = int(line[66:70])
        MF = int(line[70:72])
        MT = int(line[72:75])
        NS = int(line[75:80])
        return [HL, MAT, MF, MT, NS]

    def _get_cont_record(self, line=None, skipC=False):
        if self._veryverbose:
            print("Get CONT record")
        if not line:
            line = self._fh.readline()
        if skipC:
            C1 = None
            C2 = None
        else:
            C1 = endftod(line[:11])
            C2 = endftod(line[11:22])
        L1 = int(line[22:33])
        L2 = int(line[33:44])
        N1 = int(line[44:55])
        N2 = int(line[55:66])
        MAT = int(line[66:70])
        MF = int(line[70:72])
        MT = int(line[72:75])
        NS = int(line[75:80])
        return [C1, C2, L1, L2, N1, N2, MAT, MF, MT, NS]

    def _get_head_record(self, line=None):
        if not line:
            line = self._fh.readline()
        if self._veryverbose:
            print("Get HEAD record")
        ZA = int(endftod(line[:11]))
        AWR = endftod(line[11:22])
        L1 = int(line[22:33])
        L2 = int(line[33:44])
        N1 = int(line[44:55])
        N2 = int(line[55:66])
        MAT = int(line[66:70])
        MF = int(line[70:72])
        MT = int(line[72:75])
        NS = int(line[75:80])
        return [ZA, AWR, L1, L2, N1, N2, MAT, MF, MT, NS]

    def _get_list_record(self, onlyList=False):
        # determine how many items are in list
        if self._veryverbose:
            print("Get LIST record")
        items = self._get_cont_record()
        NPL = items[4]

        # read items
        itemsList = []
        m = 0
        for i in range((NPL-1)//6 + 1):
            line = self._fh.readline()
            toRead = min(6,NPL-m)
            for j in range(toRead):
                val = endftod(line[0:11])
                itemsList.append(val)
                line = line[11:]
            m = m + toRead
        if onlyList:
            return itemsList
        else:
            return (items, itemsList)

    def _get_tab1_record(self):
        if self._veryverbose:
            print("Get TAB1 record")
        return Tab1.from_file(self._fh)

    def _get_tab2_record(self):
        if self._veryverbose:
            print("Get TAB2 record")
        r = ENDFTab2Record()
        r.read(self._fh)
        return r

    def _seek_mfmt(self, MF, MT):
        self._fh.seek(self._start_position)
        searchString = '{0:4}{1:2}{2:3}'.format(self.material, MF, MT)
        while True:
            position = self._fh.tell()
            line = self._fh.readline()
            if line == '':
                # Reached EOF
                if self._verbose:
                    print("Could not find MF={0}, MT={1}".format(MF, MT))
                raise NotFound('Reaction')
            if line[66:75] == searchString:
                self._fh.seek(position)
                break

    def _print_info(self, MF, MT):
        if self._verbose:
            print("Reading MF={0}, MT={1} {2}".format(MF, MT, label(MT)))

    def __repr__(self):
        try:
            name = libraries[self.files[0].NLIB]
            nuclide = self.files[0].ZA
        except:
            name = "Undetermined"
            nuclide = "None"
        return "<{0} Evaluation: {1}>".format(name, nuclide)


class Tab1(object):
    def __init__(self, nbt, interp, x, y):
        """Create an ENDF Tab1 object."""

        if len(nbt) == 0 and len(interp) == 0:
            self.n_regions = 1
            self.nbt = np.array([len(x)])
            self.interp = np.array([2])  # linear-linear by default
        else:
            # NR=0 implies linear-linear interpolation by default
            self.n_regions = len(nbt)
            self.nbt = np.asarray(nbt, dtype=int)
            self.interp = np.asarray(interp, dtype=int)

        self.n_pairs = len(x)
        self.x = np.asarray(x)  # Abscissa values
        self.y = np.asarray(y)  # Ordinate values

    @classmethod
    def from_ndarray(cls, array, idx=0):
        # Get number of regions and pairs
        n_regions = int(array[idx])
        n_pairs = int(array[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            nbt = np.asarray(array[idx:idx + n_regions], dtype=int)
            interp = np.asarray(array[idx + n_regions:idx + 2*n_regions], dtype=int)
        else:
            # Zero regions implies linear-linear interpolation by default
            nbt = np.array([n_pairs])
            interp = np.array([2])

        # Get (x,y) pairs
        idx += 2*n_regions + 1
        x = array[idx:idx + n_pairs]
        y = array[idx + n_pairs:idx + 2*n_pairs]

        return cls(nbt, interp, x, y)

    @classmethod
    def from_file(cls, fh):
        # Determine how many interpolation regions and total points there are
        line = fh.readline()
        C1 = endftod(line[:11])
        C2 = endftod(line[11:22])
        L1 = int(line[22:33])
        L2 = int(line[33:44])
        n_regions = int(line[44:55])
        n_pairs = int(line[55:66])
        params = [C1, C2, L1, L2]

        # Read the interpolation region data, namely NBT and INT
        nbt = np.zeros(n_regions)
        interp = np.zeros(n_regions)
        m = 0
        for i in range((n_regions - 1)//3 + 1):
            line = fh.readline()
            toRead = min(3, n_regions - m)
            for j in range(toRead):
                nbt[m] = int(line[0:11])
                interp[m] = int(line[11:22])
                line = line[22:]
                m += 1

        # Read tabulated pairs x(n) and y(n)
        x = np.zeros(n_pairs)
        y = np.zeros(n_pairs)
        m = 0
        for i in range((n_pairs - 1)//3 + 1):
            line = fh.readline()
            toRead = min(3, n_pairs - m)
            for j in range(toRead):
                x[m] = endftod(line[:11])
                y[m] = endftod(line[11:22])
                line = line[22:]
                m += 1

        return params, cls(nbt, interp, x, y)


class ENDFTab2Record(object):
    def __init__(self):
        self.NBT = []
        self.INT = []

    def read(self, fh):
        # Determine how many interpolation regions and total points there are
        line = fh.readline()
        C1 = endftod(line[:11])
        C2 = endftod(line[11:22])
        L1 = int(line[22:33])
        L2 = int(line[33:44])
        NR = int(line[44:55])
        NZ = int(line[55:66])
        self.params = [C1, C2, L1, L2, NR, NZ]

        # Read the interpolation region data, namely NBT and INT
        m = 0
        for i in range((NR-1)//3 + 1):
            line = fh.readline()
            toRead = min(3,NR-m)
            for j in range(toRead):
                NBT = int(line[0:11])
                INT = int(line[11:22])
                self.NBT.append(NBT)
                self.INT.append(INT)
                line = line[22:]
            m = m + toRead


class EnergyDistribution(object):
    def __init__(self):
        pass

class Reaction(object):
    """A single MT record on an ENDF file."""

    def __init__(self, MT):
        self.MT = MT
        self.MFs = []

    def __repr__(self):
        return "<ENDF Reaction: MT={0}, {1}>".format(self.MT, label(self.MT))


class Resonance(object):
    def __init__(self):
        pass


class BreitWigner(Resonance):
    def __init__(self):
        pass

    def __repr__(self):
        return "<Breit-Wigner Resonance: l={0.L} J={0.J} E={0.E}>".format(self)


class ReichMoore(Resonance):
    def __init__(self):
        pass

    def __repr__(self):
        return "<Reich-Moore Resonance: l={0.L} J={0.J} E={0.E}>".format(self)


class AdlerAdler(Resonance):
    def __init__(self):
        pass


class RMatrixLimited(Resonance):
    def __init__(self):
        pass


class NotFound(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
