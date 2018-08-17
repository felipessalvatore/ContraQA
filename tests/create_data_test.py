import unittest
import os
import shutil
from contra_qa.text_generation.boolean1_neg import boolean1


class AddDataset(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("data"):
            shutil.rmtree("data")

    @classmethod
    def setUp(cls):
        pass

    def test_generate_boolean1(self):
        boolean1()
        cond = os.path.join("data", "boolean1_test.csv")
        cond = cond and os.path.join("data", "boolean1_train.csv")
        self.assertTrue(cond)
