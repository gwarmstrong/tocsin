#distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "../fastp-bind/src/api.h":
    vector[string] fastq_reader(string fastq_filename)
