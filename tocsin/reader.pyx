from reader cimport fastq_reader, fastq_encoder

def fq_reader(string fastq_filename):

    results = fastq_reader(fastq_filename)

    return results

def fq_encoder(string fastq_filename, int length=150):

    results = fastq_encoder(fastq_filename, length)

    return results

def fq_encoder_alt(string fastq_filename, int length=150):

    results = fastq_encoder_alt(fastq_filename, length)

    return results
