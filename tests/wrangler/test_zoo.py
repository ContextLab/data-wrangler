#!/usr/bin/env python

"""
Tests for `datawrangler` package (format module).

Backend Testing Strategy:
-------------------------
This test suite uses pytest parameterization to test all functions with both
pandas and Polars backends. Key testing principles:

1. Data Equivalence: Both backends should produce equivalent data values
2. Backend-Specific Behavior: Some features (like index names) behave differently
3. Performance Testing: Polars should be faster, but we primarily test correctness
4. Cross-Backend Equivalence: Deterministic operations should produce identical results

Backend Differences Tested:
- pandas: Preserves index names and metadata during operations
- Polars: Uses position-based indexing, may lose index names during conversion
- ML Models: Random seed management ensures deterministic cross-backend results
"""

import datawrangler as dw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from .conftest import assert_backend_type, assert_dataframes_equivalent


def test_is_dataframe(data, img_file, text_file):
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(pd.DataFrame(np.zeros([10, 3])))
    assert not dw.zoo.is_dataframe(img_file)
    assert not dw.zoo.is_dataframe(text_file)


def test_dataframe_like(data, img_file):
    assert dw.zoo.dataframe_like(data)
    assert not dw.zoo.dataframe_like(img_file)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_dataframe(data, data_file, backend):
    # Test basic DataFrame wrangling with backend
    df = dw.zoo.wrangle_dataframe(data, backend=backend)
    assert dw.zoo.is_dataframe(df)
    assert_backend_type(df, backend)

    # Test with pandas input and specified backend
    df1 = dw.zoo.wrangle_dataframe(data, backend=backend)
    
    # Convert to pandas for detailed assertions (values should be equivalent)
    df1_pandas = df1.to_pandas() if isinstance(df1, pl.DataFrame) else df1
    
    # Backend-specific behavior: pandas preserves index names, Polars does not
    # This is expected due to fundamental differences in how the libraries handle row indexing
    if backend == 'pandas':
        assert df1_pandas.index.name == 'ByTwos', "pandas backend should preserve index names"
    else:  # backend == 'polars'
        # Polars backend: index names are not preserved during conversion
        # This is documented behavior - Polars uses position-based indexing only
        pass
    
    assert np.all(df1_pandas['FirstDim'] == np.arange(1, 8))
    assert np.all(df1_pandas['SecondDim'] == np.arange(2, 16, 2))
    assert np.all(df1_pandas['ThirdDim'] == np.arange(3, 24, 3))
    assert np.all(df1_pandas['FourthDim'] == np.arange(4, 32, 4))
    assert np.all(df1_pandas['FifthDim'] == np.arange(5, 40, 5))

    # Test loading from file with backend
    df2 = dw.zoo.wrangle_dataframe(data_file, load_kwargs={'index_col': 0}, backend=backend)
    assert_backend_type(df2, backend)
    
    # Verify equivalence between backends
    assert_dataframes_equivalent(df1, df2)


def test_wrangle_dataframe_cross_backend_equivalence(data):
    """Test that pandas and Polars backends produce equivalent results."""
    pandas_df = dw.zoo.wrangle_dataframe(data, backend='pandas')
    polars_df = dw.zoo.wrangle_dataframe(data, backend='polars')
    
    assert_dataframes_equivalent(pandas_df, polars_df)


def test_is_array(data, img_file, text_file, data_file):
    assert dw.zoo.is_array(data.values)
    assert dw.zoo.is_array(img_file)
    assert not dw.zoo.is_array(text_file)
    assert not dw.zoo.is_array(data_file)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_array(data, img_file, backend):
    # Test array wrangling with different backends
    df = dw.zoo.wrangle_array(data.values, backend=backend)
    assert dw.zoo.is_dataframe(df)
    assert_backend_type(df, backend)
    assert df.shape == (7, 5)

    # Test image array wrangling
    df_img = dw.zoo.wrangle_array(img_file, backend=backend)
    assert df_img.shape == (1400, 5760)
    assert dw.zoo.is_dataframe(df_img)
    assert_backend_type(df_img, backend)
    
    # Convert to numpy for value checks (works for both backends)
    df_img_values = df_img.to_numpy() if hasattr(df_img, 'to_numpy') else df_img.values
    assert np.max(df_img_values) >= 245 # needed for GitHub actions tests
    assert np.min(df_img_values) == 12
    assert np.isclose(np.mean(df_img_values), 152.19, atol=0.1)


def test_wrangle_array_cross_backend_equivalence(data):
    """Test that array wrangling produces equivalent results across backends."""
    pandas_df = dw.zoo.wrangle_array(data.values, backend='pandas')
    polars_df = dw.zoo.wrangle_array(data.values, backend='polars')
    
    assert_dataframes_equivalent(pandas_df, polars_df)


def test_load_text(text_file):
    text = dw.io.load(text_file).split('\n')
    assert text[0] == 'O give me a home where the buffaloes roam'
    assert text[-1] == 'And the skies are not cloudy all day'


def test_is_text(text_file, img_file, data_file):
    assert dw.zoo.is_text(text_file)
    assert not dw.zoo.is_text(img_file)
    assert not dw.zoo.is_text(data_file)


def test_get_corpus():
    # test sotus corpus (small)
    sotus = dw.zoo.text.get_corpus('sotus')
    assert sotus[0].split('\n')[0] == 'Mr. Speaker, Mr. President, and distinguished Members of the House and ' \
                                      'Senate, honored guests, and fellow citizens:'
    assert sotus[0].split('\n')[-1] == 'Thank you. God bless you, and God bless America.'
    assert sotus[-1].split('\n')[0] == "Thank you very much. Mr. Speaker, Mr. Vice President, Members of Congress, " \
                                       "the First Lady of the United States, and citizens of America: Tonight, " \
                                       "as we mark the conclusion of our celebration of Black History Month, " \
                                       "we are reminded of our Nation's path towards civil rights and the work that " \
                                       "still remains to be done. Recent threats targeting Jewish community centers " \
                                       "and vandalism of Jewish cemeteries, as well as last week's shooting in " \
                                       "Kansas City, remind us that while we may be a nation divided on policies, " \
                                       "we are a country that stands united in condemning hate and evil in all of " \
                                       "its very ugly forms."
    assert sotus[-1].split('\n')[-1] == ''
    assert len(sotus) == 29

    # test small hugging face corpus: cbt/raw
    cbt = dw.zoo.text.get_corpus('cbt', 'raw')
    assert cbt[0][:100] == 'CHAPTER I. -LCB- Chapter heading picture : p1.jpg -RCB- How the Fairies were not Invited ' \
                           'to Court . '
    assert cbt[0][-104:] == "occasionally Rosalind would say , `` I do believe , my dear , that you are really as " \
                            "clever as ever ! ''"
    assert len(cbt[0]) == 98440
    assert len(cbt[100]) == 417432
    assert len(cbt) == 108


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_text_sklearn(text_file, backend):
    text = dw.io.load(text_file).split('\n')

    # scikit-learn CountVectorizer with backend
    text_kwargs = {'model': 'CountVectorizer'}
    cv = dw.wrangle(text, text_kwargs=text_kwargs, backend=backend)
    assert cv.shape == (24, 1220)
    assert_backend_type(cv, backend)
    
    # Convert to values for numerical checks
    cv_values = cv.to_numpy() if hasattr(cv, 'to_numpy') else cv.values
    assert dw.util.btwn(cv_values, 0, 1)

    # check text split into two documents
    i = 10
    cv2 = dw.wrangle([text[:i], text[i:]], text_kwargs=text_kwargs, backend=backend)
    cv2_part = cv2[1].to_numpy() if hasattr(cv2[1], 'to_numpy') else cv2[1].values
    cv_part = cv_values[i:]
    assert np.allclose(cv_part, cv2_part)

    # scikit-learn CountVectorizer + LatentDirichletAllocation
    text_kwargs = {'model': ['CountVectorizer', 'LatentDirichletAllocation'], 'corpus': 'sotus'}
    lda = dw.wrangle(text, text_kwargs=text_kwargs, backend=backend)
    assert lda.shape == (24, 50)
    assert_backend_type(lda, backend)
    
    lda_values = lda.to_numpy() if hasattr(lda, 'to_numpy') else lda.values
    assert dw.util.btwn(lda_values, 0, 1)
    assert np.allclose(lda_values.sum(axis=1), 1)

    # scikit-learn TfidfVectorizer + NMF
    text_kwargs = {'model': ['TfidfVectorizer', {'model': 'NMF', 'args': [], 'kwargs': {'n_components': 25}}],
                   'corpus': 'sotus'}
    nmf = dw.wrangle(text, text_kwargs=text_kwargs, backend=backend)
    assert nmf.shape == (24, 25)
    assert_backend_type(nmf, backend)
    
    nmf_values = nmf.to_numpy() if hasattr(nmf, 'to_numpy') else nmf.values
    assert dw.util.btwn(nmf_values, 0, 1)


def test_wrangle_text_sklearn_cross_backend_equivalence(text_file):
    """Test that sklearn text processing produces equivalent results across backends."""
    import numpy as np
    
    text = dw.io.load(text_file).split('\n')
    
    # Test CountVectorizer equivalence - this should be deterministic
    text_kwargs = {'model': 'CountVectorizer'}
    pandas_cv = dw.wrangle(text, text_kwargs=text_kwargs, backend='pandas')
    polars_cv = dw.wrangle(text, text_kwargs=text_kwargs, backend='polars')
    assert_dataframes_equivalent(pandas_cv, polars_cv)
    
    # Test LDA equivalence with fixed random seed for deterministic behavior
    # Set random seed before each backend test to ensure identical results
    np.random.seed(42)
    text_kwargs = {
        'model': [
            'CountVectorizer', 
            {'model': 'LatentDirichletAllocation', 'args': [], 'kwargs': {'random_state': 42}}
        ], 
        'corpus': 'sotus'
    }
    pandas_lda = dw.wrangle(text, text_kwargs=text_kwargs, backend='pandas')
    
    np.random.seed(42)  # Reset seed for second backend
    polars_lda = dw.wrangle(text, text_kwargs=text_kwargs, backend='polars')
    
    assert_dataframes_equivalent(pandas_lda, polars_lda)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_text_hugging_face(text_file, backend):
    text = dw.io.load(text_file).split('\n')
    words = [s.split() for s in text]

    # Test sentence transformer with different backends
    sentence_transformer_kwargs = {'model': {'model': 'all-MiniLM-L6-v2', 'args': [], 'kwargs': {}}}
    sentence_embeddings = dw.wrangle(words, text_kwargs=sentence_transformer_kwargs, backend=backend)
    assert len(sentence_embeddings) == 24
    assert all([a == b for a, b in zip([g.shape[0] for g in sentence_embeddings], [len(w) for w in words])])
    assert all(g.shape[1] == 384 for g in sentence_embeddings)  # all-MiniLM-L6-v2 produces 384-dim embeddings
    
    # Convert to values for numerical checks
    embedding_means = []
    for g in sentence_embeddings:
        g_values = g.to_numpy() if hasattr(g, 'to_numpy') else g.values
        embedding_means.append(g_values.mean())
    assert np.allclose(embedding_means,
                       [0.0011996, 0.0009824, 0.001065, 0.0012655, 0.001737, 0.0009824,
                        0.001065, 0.0012655, 0.0012792, 0.0010756, 0.0012165, 0.0014435,
                        0.001737, 0.0009824, 0.001065, 0.0012655, 0.0013562, 0.0010959,
                        0.0015061, 0.0013031, 0.001737, 0.0009824, 0.001065, 0.0012655], atol=0.001)

    # Test different transformer models
    distilbert_kwargs = {'model': {'model': 'all-mpnet-base-v2', 'args': [], 'kwargs': {}}}
    distilbert_embeddings = dw.wrangle(text, text_kwargs=distilbert_kwargs, backend=backend)
    assert distilbert_embeddings.shape == (24, 768)  # all-mpnet-base-v2 produces 768-dim embeddings
    assert_backend_type(distilbert_embeddings, backend)
    
    distilbert_values = distilbert_embeddings.to_numpy() if hasattr(distilbert_embeddings, 'to_numpy') else distilbert_embeddings.values
    assert np.isclose(distilbert_values.mean(axis=0).mean(axis=0), -0.000105, atol=0.0001)

    bert_kwargs = {'model': {'model': 'all-MiniLM-L12-v2', 'args': [], 'kwargs': {}}}
    bert_embeddings = dw.wrangle(text, text_kwargs=bert_kwargs, backend=backend)
    assert bert_embeddings.shape == (24, 384)  # all-MiniLM-L12-v2 produces 384-dim embeddings
    assert_backend_type(bert_embeddings, backend)
    
    bert_values = bert_embeddings.to_numpy() if hasattr(bert_embeddings, 'to_numpy') else bert_embeddings.values
    assert np.isclose(bert_values.mean(axis=0).mean(axis=0), -0.0001967, atol=0.0001)


def test_wrangle_text_hugging_face_cross_backend_equivalence(text_file):
    """Test that HuggingFace text processing produces equivalent results across backends."""
    text = dw.io.load(text_file).split('\n')
    
    # Test transformer model equivalence
    model_kwargs = {'model': {'model': 'all-mpnet-base-v2', 'args': [], 'kwargs': {}}}
    pandas_embeddings = dw.wrangle(text, text_kwargs=model_kwargs, backend='pandas')
    polars_embeddings = dw.wrangle(text, text_kwargs=model_kwargs, backend='polars')
    assert_dataframes_equivalent(pandas_embeddings, polars_embeddings)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_text_simplified_api(text_file, backend):
    """Test simplified text model API with backward compatibility."""
    text = dw.io.load(text_file).split('\n')
    
    # Test 1: String model format (simplified API)
    simple_string_kwargs = {'model': 'all-MiniLM-L6-v2'}
    simple_embeddings = dw.wrangle(text, text_kwargs=simple_string_kwargs, backend=backend)
    assert simple_embeddings.shape == (24, 384)  # all-MiniLM-L6-v2 produces 384-dim embeddings
    assert_backend_type(simple_embeddings, backend)
    
    # Test 2: Partial dict format (model key only)
    partial_dict_kwargs = {'model': {'model': 'all-MiniLM-L6-v2'}}
    partial_embeddings = dw.wrangle(text, text_kwargs=partial_dict_kwargs, backend=backend)
    assert partial_embeddings.shape == (24, 384)
    assert_backend_type(partial_embeddings, backend)
    
    # Test 3: Full dict format (backward compatibility)
    full_dict_kwargs = {'model': {'model': 'all-MiniLM-L6-v2', 'args': [], 'kwargs': {}}}
    full_embeddings = dw.wrangle(text, text_kwargs=full_dict_kwargs, backend=backend)
    assert full_embeddings.shape == (24, 384)
    assert_backend_type(full_embeddings, backend)
    
    # Test 4: Verify all formats produce equivalent results
    simple_values = simple_embeddings.to_numpy() if hasattr(simple_embeddings, 'to_numpy') else simple_embeddings.values
    partial_values = partial_embeddings.to_numpy() if hasattr(partial_embeddings, 'to_numpy') else partial_embeddings.values
    full_values = full_embeddings.to_numpy() if hasattr(full_embeddings, 'to_numpy') else full_embeddings.values
    
    # All three formats should produce identical results
    assert np.allclose(simple_values, partial_values, atol=1e-6)
    assert np.allclose(simple_values, full_values, atol=1e-6)
    assert np.allclose(partial_values, full_values, atol=1e-6)


def test_wrangle_text_simplified_api_cross_backend_equivalence(text_file):
    """Test that simplified API produces equivalent results across backends."""
    text = dw.io.load(text_file).split('\n')
    
    # Test string format equivalence across backends
    string_kwargs = {'model': 'all-MiniLM-L6-v2'}
    pandas_string = dw.wrangle(text, text_kwargs=string_kwargs, backend='pandas')
    polars_string = dw.wrangle(text, text_kwargs=string_kwargs, backend='polars')
    assert_dataframes_equivalent(pandas_string, polars_string)
    
    # Test partial dict format equivalence across backends
    partial_kwargs = {'model': {'model': 'all-MiniLM-L6-v2'}}
    pandas_partial = dw.wrangle(text, text_kwargs=partial_kwargs, backend='pandas')
    polars_partial = dw.wrangle(text, text_kwargs=partial_kwargs, backend='polars')
    assert_dataframes_equivalent(pandas_partial, polars_partial)


def test_normalize_text_model():
    """Test the normalize_text_model utility function."""
    # Test string normalization
    result = dw.zoo.text.normalize_text_model('all-MiniLM-L6-v2')
    expected = {'model': 'all-MiniLM-L6-v2', 'args': [], 'kwargs': {}}
    assert result == expected
    
    # Test partial dict normalization
    result = dw.zoo.text.normalize_text_model({'model': 'all-MiniLM-L6-v2'})
    expected = {'model': 'all-MiniLM-L6-v2', 'args': [], 'kwargs': {}}
    assert result == expected
    
    # Test partial dict with some args/kwargs
    result = dw.zoo.text.normalize_text_model({'model': 'all-MiniLM-L6-v2', 'args': ['arg1']})
    expected = {'model': 'all-MiniLM-L6-v2', 'args': ['arg1'], 'kwargs': {}}
    assert result == expected
    
    # Test full dict (no change)
    full_dict = {'model': 'all-MiniLM-L6-v2', 'args': [], 'kwargs': {}}
    result = dw.zoo.text.normalize_text_model(full_dict)
    assert result == full_dict
    
    # Test non-dict/non-string input (passthrough)
    result = dw.zoo.text.normalize_text_model(None)
    assert result is None


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_text_list_models_simplified_api(text_file, backend):
    """Test simplified API with lists of models."""
    text = dw.io.load(text_file).split('\n')
    
    # Test 1: List of string models (sklearn pipeline)
    sklearn_list_kwargs = {'model': ['CountVectorizer', 'LatentDirichletAllocation'], 'corpus': 'sotus'}
    sklearn_result = dw.wrangle(text, text_kwargs=sklearn_list_kwargs, backend=backend)
    assert sklearn_result.shape == (24, 50)  # LDA with default 50 topics
    assert_backend_type(sklearn_result, backend)
    
    # Test 2: Mixed list with string and dict models
    mixed_list_kwargs = {
        'model': [
            'CountVectorizer',  # String format
            {'model': 'NMF', 'args': [], 'kwargs': {'n_components': 20}}  # Dict format
        ],
        'corpus': 'sotus'
    }
    mixed_result = dw.wrangle(text, text_kwargs=mixed_list_kwargs, backend=backend)
    assert mixed_result.shape == (24, 20)  # NMF with 20 components
    assert_backend_type(mixed_result, backend)
    
    # Test 3: Verify list processing maintains backward compatibility
    old_style_kwargs = {
        'model': [
            {'model': 'TfidfVectorizer', 'args': [], 'kwargs': {}},
            {'model': 'NMF', 'args': [], 'kwargs': {'n_components': 15}}
        ],
        'corpus': 'sotus'
    }
    old_style_result = dw.wrangle(text, text_kwargs=old_style_kwargs, backend=backend)
    assert old_style_result.shape == (24, 15)
    assert_backend_type(old_style_result, backend)


def test_is_null():
    assert dw.zoo.is_null(None)
    assert dw.zoo.is_null('')
    assert dw.zoo.is_null([])
    assert dw.zoo.is_null([[]])
    assert dw.zoo.is_null([[], [[], [], [[], []]]])
    assert not dw.zoo.is_null([1, 2, 3, 4, 5])
    assert not dw.zoo.is_null('this should not be null')
    assert not dw.zoo.is_null([[], 'neither should', [['this'], []]])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_wrangle_null(backend):
    df = dw.wrangle(None, backend=backend)
    assert dw.zoo.is_dataframe(df)
    assert_backend_type(df, backend)
    assert len(df) == 0


def test_wrangle_null_cross_backend_equivalence():
    """Test that null wrangling produces equivalent results across backends."""
    pandas_df = dw.wrangle(None, backend='pandas')
    polars_df = dw.wrangle(None, backend='polars')
    
    # Both should be empty DataFrames
    assert len(pandas_df) == 0
    assert len(polars_df) == 0
    assert_dataframes_equivalent(pandas_df, polars_df)
