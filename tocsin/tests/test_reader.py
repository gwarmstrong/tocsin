from tocsin.tests.testing import TocsinTestCase
from tocsin.reader import fq_encoder, fq_reader


class TestCythonReaders(TocsinTestCase):

    package = 'tocsin.tests'

    def test_fq_reader(self):
        demo_file = self.get_data_path('small_test_fastq_01.fq')
        lines = fq_reader(bytes(demo_file, 'utf-8'))
        expected_reads = [
            b'CGAT',
            b'CGAG',
            b'CGAC',
            b'CGAC',
        ]
        self.assertListEqual(lines, expected_reads)

    def test_fq_encoder(self):
        demo_file = self.get_data_path('small_test_fastq_01.fq')
        lines = fq_encoder(bytes(demo_file, 'utf-8'), 4)
        expected_reads = [
            [1, 2, 0, 3],
            [1, 2, 0, 2],
            [1, 2, 0, 1],
            [1, 2, 0, 1],
        ]

        self.assertEqual(len(lines), len(expected_reads))
        for r1, r2 in zip(lines, expected_reads):
            self.assertListEqual(r1, r2)

    def test_fq_encoder_default_len(self):
        demo_file = self.get_data_path('small_test_fastq_01.fq')
        lines = fq_encoder(bytes(demo_file, 'utf-8'), 5)
        expected_reads = [
            [1, 2, 0, 3, -1],
            [1, 2, 0, 2, -1],
            [1, 2, 0, 1, -1],
            [1, 2, 0, 1, -1],
        ]

        self.assertEqual(len(lines), len(expected_reads))
        for r1, r2 in zip(lines, expected_reads):
            self.assertListEqual(r1, r2)

    def test_fq_encoder_default_len_and_shorter_than_longest(self):
        demo_file = self.get_data_path('small_test_fastq_02.fq')
        lines = fq_encoder(bytes(demo_file, 'utf-8'), 6)
        expected_reads = [
            # AACGT, CGGGAA, AGCGCCTT
            [0, 0, 1, 2, 3, -1],
            [1, 2, 2, 2, 0, 0],
            [0, 2, 1, 2, 1, 1],
        ]

        self.assertEqual(len(lines), len(expected_reads))
        for r1, r2 in zip(lines, expected_reads):
            self.assertListEqual(r1, r2)
