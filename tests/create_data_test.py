import unittest
import os
import shutil
from contra_qa.text_generation.boolean1_neg import boolean1
from contra_qa.text_generation.boolean2_S_and import boolean2
from contra_qa.text_generation.boolean3_NP_and import boolean3
from contra_qa.text_generation.boolean4_VP_and import boolean4
from contra_qa.text_generation.boolean5_AP_and import boolean5
from contra_qa.text_generation.boolean6_implicit_and import boolean6
from contra_qa.text_generation.boolean7_S_or import boolean7
from contra_qa.text_generation.boolean8_NP_or import boolean8
from contra_qa.text_generation.boolean9_VP_or import boolean9
from contra_qa.text_generation.boolean10_AP_or import boolean10
from contra_qa.text_generation.boolean_data_gen import create_all



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

    def test_generate_boolean6(self):
        boolean6()
        path_train = os.path.join("data", "boolean6_train.csv")
        path_test = os.path.join("data", "boolean6_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean7(self):
        boolean7()
        path_train = os.path.join("data", "boolean7_train.csv")
        path_test = os.path.join("data", "boolean7_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean8(self):
        boolean8()
        path_train = os.path.join("data", "boolean8_train.csv")
        path_test = os.path.join("data", "boolean8_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean9(self):
        boolean9()
        path_train = os.path.join("data", "boolean9_train.csv")
        path_test = os.path.join("data", "boolean9_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_boolean10(self):
        boolean10()
        path_train = os.path.join("data", "boolean10_train.csv")
        path_test = os.path.join("data", "boolean10_test.csv")
        cond = os.path.exists(path_train) and os.path.exists(path_test)
        self.assertTrue(cond)

    def test_generate_all_boolean_data(self):
        create_all()
        path_train_AND = os.path.join("data", "boolean_AND_train.csv")
        path_test_AND = os.path.join("data", "boolean_AND_test.csv")
        path_train_OR = os.path.join("data", "boolean_OR_train.csv")
        path_test_OR = os.path.join("data", "boolean_OR_test.csv")
        path_train_ALL = os.path.join("data", "boolean_train.csv")
        path_test_ALL = os.path.join("data", "boolean_test.csv")
        cond_AND = os.path.exists(path_train_AND) and os.path.exists(path_test_AND) # noqa
        cond_OR = os.path.exists(path_train_OR) and os.path.exists(path_test_OR) # noqa
        cond_ALL = os.path.exists(path_train_ALL) and os.path.exists(path_test_ALL) # noqa
        self.assertTrue(cond_AND)
        self.assertTrue(cond_OR)
        self.assertTrue(cond_ALL)
