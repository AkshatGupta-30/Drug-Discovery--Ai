import pytest
from drug_generator.config import config
from drug_generator.preprocessing.data_management import load_dataset
import drug_generator.preprocessing.preprocessors as pp
from drug_generator.predict import generate_molecule
import random


@pytest.fixture
def single_molecule_generation():
    cv_data=load_dataset(config.TEST_FILE)
    preprocess = pp.preprocess_data()
    preprocess.fit()
    X_cv,_ = preprocess.transform(cv_data["X"],cv_data["Y"])
    single_input = X_cv[random.randint(0,X_cv.shape[0]),:]
    single_input= single_input.reshape(1,single_input.shape[0])

    gen_mol = generate_molecule(single_input,1)
    print(gen_mol)
    return gen_mol


def test_is_molecule_generation_none(single_molecule_generation):
    assert single_molecule_generation is not None


def test_gen_molcule_dtype(single_molecule_generation):
    assert isinstance(single_molecule_generation.get("Generated Valid Molecule"),str)

