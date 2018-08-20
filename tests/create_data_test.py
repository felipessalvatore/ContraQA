import unittest
import os
import shutil
from contra_qa.text_generation.boolean1_neg import boolean1
from contra_qa.text_generation.boolean2_S_and import boolean2
from contra_qa.text_generation.boolean3_NP_and import boolean3
from contra_qa.text_generation.boolean4_VP_and import boolean4
from contra_qa.text_generation.boolean5_AP_and import boolean5


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
        path_train = os.path.join("data", "boolean1_train.csv")
        path_test = os.path.join("data", "boolean1_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean2(self):
        boolean2()
        path_train = os.path.join("data", "boolean2_train.csv")
        path_test = os.path.join("data", "boolean2_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean3(self):
        boolean3()
        path_train = os.path.join("data", "boolean3_train.csv")
        path_test = os.path.join("data", "boolean3_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean4(self):
        boolean4()
        path_train = os.path.join("data", "boolean4_train.csv")
        path_test = os.path.join("data", "boolean4_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean5(self):
        boolean5()
        path_train = os.path.join("data", "boolean5_train.csv")
        path_test = os.path.join("data", "boolean5_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)
