import unittest
from os import getcwd
from os.path import split
from helper_functions import translate_to_local_file_path

class HelperFunctionsTestCase(unittest.TestCase):
    """Test for HelperFunction.py"""
    
    def test_translate_to_local_file_path(self):
        pwd=getcwd()
        head, tail = split(pwd)
        filename= "test.txt"
        dirname="dir"
        file_path = translate_to_local_file_path(filename)
        self.assertEqual(file_path, f"file:///{head}/{filename}")
        file_path = translate_to_local_file_path(filename,dirname)
        self.assertEqual(file_path, f"file:///{head}/{dirname}/{filename}")


if __name__ == "__main__":
    unittest.main()