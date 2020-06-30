import tempfile
import json
import io

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

from science_tagger.utils import (load_data, calc_performance_per_tag,
    load_test_data, yield_train_data, load_train_test_data)


def test_load_data():    
    data = [
        {
            'text': 'A',
            'tags': ['T1', 'T2'],
            'meta': {'Grant_ID': 1, 'Title': 'A'}
        }
    ]
    with tempfile.NamedTemporaryFile('w') as tmp:
        for line in data:
            tmp.write(json.dumps(line))
            tmp.write('\n')
        tmp.seek(0)
        texts, tags, meta = load_data(tmp.name)
    assert len(texts) == 1
    assert len(tags) == 1
    assert len(meta) == 1
    assert texts == ['A']
    assert tags == [['T1', 'T2']]
    assert meta == [{'Grant_ID': 1, 'Title': 'A'}]

def test_load_data_with_label_binarizer():
    data = [
        {
            'text': 'A',
            'tags': ['T1', 'T2'],
            'meta': {'Grant_ID': 1, 'Title': 'A'}
        },
        {
            'text': 'B',
            'tags': ['T1'],
            'meta': {'Grant_ID': 2, 'Title': 'B'}
        }
    ]
    with tempfile.NamedTemporaryFile('w') as tmp:
        for line in data:
            tmp.write(json.dumps(line))
            tmp.write('\n')
        tmp.seek(0)
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([['T1','T2']])
        texts, tags, meta = load_data(tmp.name, label_binarizer)
    assert np.array_equal(tags[0], [1, 1])
    assert np.array_equal(tags[1], [1, 0])

def test_calc_performance_per_tag():
    Y_true = np.array([[1,0,0], [1,1,0], [0,0,1]])
    Y_pred = np.array([[0,0,1], [1,0,1], [0,0,1]])
    tags = ['T1', 'T2', 'T3']

    performance_per_tag_test = pd.DataFrame({
        'Tag': tags,
        'f1': [
            f1_score(Y_true[:,i], Y_pred[:,i])
            for i in range(Y_true.shape[1])
        ]
    })
    performance_per_tag = calc_performance_per_tag(Y_true, Y_pred, tags)
    print(performance_per_tag)
    assert performance_per_tag.equals(performance_per_tag_test)

def test_load_train_test_data():
    data = [
        {
            'text': 'A',
            'tags': ['T1', 'T2'],
            'meta': {'Grant_ID': 1, 'Title': 'A'}
        },
        {
            'text': 'B',
            'tags': ['T1'],
            'meta': {'Grant_ID': 2, 'Title': 'B'}
        },
        {
            'text': 'C',
            'tags': ['T2'],
            'meta': {'Grant_ID': 3, 'Title': 'C'}
        }
    ]
    with tempfile.NamedTemporaryFile('w') as tmp:
        for line in data:
            tmp.write(json.dumps(line))
            tmp.write('\n')
        tmp.seek(0)
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([['T1','T2']])
        texts_train, texts_test, tags_train, tags_test = load_train_test_data(tmp.name, label_binarizer)
    assert np.array_equal(tags_train[0], [1, 0])
    assert np.array_equal(tags_train[1], [0, 1])
    assert np.array_equal(tags_test[0], [1, 1])

def test_yield_train_data():
    data = [
        {
            'text': 'A',
            'tags': ['T1', 'T2'],
            'meta': {'Grant_ID': 1, 'Title': 'A'}
        },
        {
            'text': 'B',
            'tags': ['T1'],
            'meta': {'Grant_ID': 2, 'Title': 'B'}
        }
    ]
    with tempfile.NamedTemporaryFile('w') as tmp:
        for line in data:
            tmp.write(json.dumps(line))
            tmp.write('\n')
        tmp.seek(0)
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([['T1','T2']])
        texts = []
        tags = []
        for text, tag in yield_train_data(tmp.name, label_binarizer):
            texts.extend(text)
            tags.extend(tag)
    assert np.array_equal(tags[0], [1, 1])
    assert np.array_equal(tags[1], [1, 0])

def test_load_test_data():
    data = [
        {
            'text': 'A',
            'tags': ['T1','T2'],
            'meta': {'Grant_ID': 1, 'Title': 'A'}
        },
        {
            'text': 'B',
            'tags': ['T1'],
            'meta': {'Grant_ID': 2, 'Title': 'B'}
        }
    ]
    with tempfile.NamedTemporaryFile('w') as tmp:
        for line in data:
            tmp.write(json.dumps(line))
            tmp.write('\n')
        tmp.seek(0)
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([['T1','T2']])
        texts, tags = load_test_data(tmp.name, label_binarizer)
    assert np.array_equal(tags[0], [1, 1])
    assert np.array_equal(tags[1], [1, 0])
