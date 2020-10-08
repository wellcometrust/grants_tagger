import pytest
import numpy as np

MESH_TAGS = [
    "D008288", # Malaria
    "D006678" # HIV
]
X = [
    "Malaria is a mosquito-borne infectious disease that affects humans and other animals. Malaria causes symptoms that typically include fever, tiredness, vomiting, and headaches. In severe cases it can cause yellow skin, seizures, coma, or death",
    "The human immunodeficiency viruses are two species of Lentivirus that infect humans. Without treatment, average survival time after infection with HIV is estimated to be 9 to 11 years, depending on the HIV subtype."
]

@pytest.mark.scispacy
def test_fit():
    from grants_tagger.scispacy_meshtagger import SciSpacyMeshTagger
    scispacy_meshtagger = SciSpacyMeshTagger(mesh_tags = MESH_TAGS)
    scispacy_meshtagger.fit()

@pytest.mark.scispacy
def test_predict():
    from grants_tagger.scispacy_meshtagger import SciSpacyMeshTagger
    scispacy_meshtagger = SciSpacyMeshTagger(mesh_tags = MESH_TAGS)
    scispacy_meshtagger.fit()
    Y_pred = scispacy_meshtagger.predict(X)
    assert Y_pred[0, 0] == 1
    assert Y_pred[1, 1] == 1

@pytest.mark.scispacy
def test_score():
    from grants_tagger.scispacy_meshtagger import SciSpacyMeshTagger
    scispacy_meshtagger = SciSpacyMeshTagger(mesh_tags = MESH_TAGS)
    scispacy_meshtagger.fit()
    Y = np.array([[1, 0], [0,1]])
    score = scispacy_meshtagger.score(X, Y)
    assert score == 1
