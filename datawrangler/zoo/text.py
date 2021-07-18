import six
import os
import numpy as np
from flair.data import Sentence
from flair.datasets import UD_ENGLISH
from flair import embeddings
from datasets import load_dataset, get_dataset_config_names, list_datasets
from sklearn.feature_extraction import text
from sklearn import decomposition

from .array import is_array, wrangle_array
from .dataframe import is_dataframe
from .null import is_null

from ..core.configurator import get_default_options, apply_defaults, update_dict
from ..io import load
from ..io.io import get_extension

defaults = get_default_options()
preloaded_corpora = {}


def is_sklearn_model(x):
    """
    Determine whether an object seems to be a valid scikit-learn model

    Parameters
    ----------
    x: the object to test

    Returns
    -------
    True if x contains "transform", "fit", and "fit_transform" methods and False otherwise.
    """
    return hasattr(x, 'transform') and hasattr(x, 'fit') and hasattr(x, 'fit_transform')


def is_hugging_face_model(x):
    """
    Determine whether an object seems to be a valid hugging-face model

    Parameters
    ----------
    x: the object to test

    Returns
    -------
    True if x contains an "embed" method, and False otherwise.
    """
    return hasattr(x, 'embed')


def robust_is_sklearn_model(x):
    """
    Wrapper for is_sklearn_model that also supports strings-- e.g., the string 'SparsePCA' will be a valid scikit-learn
    model when checked with this function, because 'SparsePCA' is defined in the sklearn.decomposition module.

    Parameters
    ----------
    x: a to-be-tested model object or a string

    Returns
    -------
    True if x (or the scikit-learn module x evaluates to) contains "transform", "fit", and "fit_transform" methods and
    False otherwise.
    """
    x = get_text_model(x)
    return is_sklearn_model(x)


def robust_is_hugging_face_model(x):
    """
    Wrapper for is_hugging_face_model that also supports strings-- e.g., the string 'WordEmbeddings' will be a valid
    hugging-face model when checked with this function, because 'WordEmbeddings' is defined in the flair.embeddings
    module and contains an "embed" method.
    ----------
    x: a to-be-tested model object or a string

    Returns
    -------
    True if x (or the hugging-face module x evaluates to) contains an "embed" method and False otherwise.
    """
    x = get_text_model(x)
    return is_hugging_face_model(x)


def get_text_model(x):
    """
    Given an valid scikit-learn or hugging-face model, or a string (e.g., 'LatentDirichletAllocation' or
    'TransformerDocumentEmbeddings') matching the name of a valid scikit-learn or hugging-face model, return
    a callable function or class constructor for the given model.

    Parameters
    ----------
    x: an object to turn into a valid scikit-learn or hugging-face model (e.g., an already-valid model or a string)

    Returns
    -------
    A valid scikit-learn or hugging-face model (or None if no model matching the given description can be found)

    """
    if is_sklearn_model(x) or is_hugging_face_model(x):
        return x  # already a valid model

    if type(x) is dict:
        if hasattr(x, 'model'):
            return get_text_model(x['model'])
        else:
            return None

    # noinspection PyShadowingNames
    def model_lookup(model_name, parent):
        try:
            return eval(f'{parent}.{model_name}')
        except AttributeError:
            return None

    for p in ['text', 'decomposition', 'embeddings']:
        m = model_lookup(x, p)
        if m is not None:
            return m
    return None


def get_corpus(dataset_name='wikipedia', config_name='20200501.en'):
    """
    Download (and return) a text corpus.  By default, a 2020 snapshot of all English Wikipedia articles is returned.

    Parameters
    ----------
    dataset_name: a string containing the corpus name.  Can be one of the following:
      - Corpora built into data-wrangler:
        - 'minipedia': a curated and cleaned up subset of Wikipedia containing articles on a wide variety of topics
        - 'neurips': a collection of NeurIPS articles
        - 'sotus': transcripts of state of the union addresses from US Presidents from 1989 -- 2018
        - 'khan': transcripts of (most) Khan Academy YouTube videos
      - Any hugging-face corpus; for a full list see https://huggingface.co/datasets
        Note that downloading hugging-face corpora also requires specifying a config_name
    config_name: configuration name or description for hugging-face corpora.  This argument is ignored if dataset_name
      is set to one of the data-wrangler corpora described above.

    Returns
    -------
    A list of number-of-documents strings, where each string contains the text of one document in the corpus.
    """

    key = f'{dataset_name}[{config_name}]'
    if key in preloaded_corpora.keys():
        return preloaded_corpora[key]

    def get_formatter(s):
        return s[s.find('_'):(s.rfind('_') + 1)]

    # built-in corpora
    corpora = {
        'minipedia': '1mRNAZlTbZzSvV3tAQfSjNm587xdYKVkX',
        'neurips': '1Qo61vh2P3Rpb9PM1lyXb5M2iw7uB03uY',
        'sotus': '1uKJtxs-C0KDM2my0K6W2p0jCF6howg1y',
        'khan': '1KPhKxQlQrZHSPlCgky7K2bsfHlvJK039'}

    if dataset_name in corpora.keys():
        print(f'loading corpus: {dataset_name}', end='...')
        corpus = load(corpora[dataset_name], dtype='numpy')['corpus']
        print('done!')

        preloaded_corpora[key] = corpus
        return corpus

    # Hugging-Face Corpus
    try:
        data = load_dataset(dataset_name, config_name)
    except FileNotFoundError:
        raise RuntimeError(f'Corpus not found: {dataset_name}.  Available corpora: {", ".join(list_datasets())}')
    except ValueError:
        raise RuntimeError(f'Configuration for {dataset_name} corpus not found: {config_name}. '
                           f'Available configurations: {", ".join(get_dataset_config_names(data_name))}')

    corpus = []
    content_keys = ['text', 'content']

    for k in data.keys():
        for c in content_keys:
            if c in data[k].data.column_names:
                for document in data[k].data[c]:
                    corpus.append(' '.join([w if '_' not in w else w.replace(get_formatter(w), ' ')
                                            for w in str(document).split()]))
    return corpus


# noinspection PyShadowingNames
def apply_text_model(x, text, *args, mode='fit_transform', return_model=False, **kwargs):
    if type(x) is list:
        models = []
        for i, v in enumerate(x):
            if (i < len(x) - 1) and ('transform' not in mode):
                temp_mode = 'fit_transform'
            else:
                temp_mode = mode

            text, m = apply_text_model(v, text, *args, mode=temp_mode, return_model=True, **kwargs)
            models.append(m)

        if return_model:
            return text, models
        else:
            return text
    elif type(x) is dict:
        assert all([k in x.keys() for k in ['model', 'args', 'kwargs']]), ValueError(f'invalid model: {x}')
        return apply_text_model(x['model'], text, *[*x['args'], *args], mode=mode, return_model=return_model,
                                **update_dict(x['kwargs'], kwargs))

    model = get_text_model(x)
    if model is None:
        raise RuntimeError(f'unsupported text processing module: {x}')

    # noinspection DuplicatedCode
    if is_sklearn_model(model):
        assert mode in ['fit', 'transform', 'fit_transform']

        if callable(model):
            model = apply_defaults(model)(*args, **kwargs)

        m = getattr(model, mode)
        transformed_text = m(text)
        if return_model:
            return transformed_text, {'model': model, 'args': args, 'kwargs': kwargs}
        return transformed_text
    elif is_hugging_face_model(model):
        if mode == 'fit':  # do nothing-- just return the un-transformed text and original model
            if return_model:
                return text, {'model': model, 'args': args, 'kwargs': kwargs}
            return text

        embedding_kwargs = kwargs.pop('embedding_kwargs', {})

        model = apply_defaults(model)(*args, **embedding_kwargs)
        wrapped_text = Sentence(text, **kwargs)
        model.embed(wrapped_text)

        # document-level embeddings-- re-compute by token
        if hasattr(wrapped_text, 'embedding') and len(wrapped_text.embedding) > 0:
            embeddings = np.empty([len(wrapped_text), len(wrapped_text.embedding)])
            embeddings[:] = np.nan

            for i, token in enumerate(wrapped_text):
                next_wrapped = Sentence(token.text)
                model.embed(next_wrapped)
                embeddings[i, :] = next_wrapped.embedding.detach().numpy()
        else:  # token-level embeddings; wrangle into an array
            embeddings = np.empty([len(wrapped_text), len(wrapped_text[0].embedding)])
            embeddings[:] = np.nan
            for i, token in enumerate(wrapped_text):
                if len(token.embedding) > 0:
                    embeddings[i, :] = token.embedding

        if return_model:
            return embeddings, {'model': model, 'args': args,
                                'kwargs': {'embedding_kwargs': embedding_kwargs,
                                           **kwargs}}
        else:
            return embeddings
    else:                                 # unknown model
        raise RuntimeError('Cannot apply text model: {model}')


def get_text(x, force_literal=False):
    if type(x) == list:
        return [get_text(t) for t in x]
    if (type(x) in six.string_types) or (type(x) == np.str_):
        if os.path.exists(x):
            if not force_literal:
                return get_text(load(x), force_literal=True)
        return x
    return None


def is_text(x):
    if type(x) == list:
        return all([is_text(i) for i in x])
    return get_text(x) is not None


def to_str_list(x, encoding='utf-8'):
    def to_string(s):
        if type(s) == str:
            return s
        elif is_null(s):
            return ''
        elif type(s) in [bytes, np.bytes_]:
            return s.decode(encoding)
        elif is_array(s) or is_dataframe(s) or (type(s) == list):
            if len(s) == 1:
                return to_string(s[0])
            else:
                return to_str_list(s, encoding=encoding)
        else:
            return str(s)

    if is_array(x) or (type(x) == list):
        return [to_string(s) for s in x]
    elif is_text(x):
        return [x]
    else:
        raise Exception('Unsupported data type: {type(x)}')


# noinspection PyShadowingNames
def wrangle_text(text, return_model=False, **kwargs):
    text = get_text(text)
    if type(text) is not list:
        text = [text]

    model = kwargs.pop('model', defaults['text']['model'])
    corpus = kwargs.pop('corpus', None)
    config = kwargs.pop('config', None)
    array_args = kwargs.pop('array_args', {})

    if type(model) is not list:
        model = [model]

    if any(robust_is_sklearn_model(m) for m in model):
        if corpus is not None:
            if not ((type(corpus) is list) and is_text(corpus)):
                corpus = get_corpus(dataset_name=corpus, config_name=config)
        else:
            corpus = get_corpus(dataset_name=defaults['text']['corpus'],
                                config_name=defaults['text']['corpus_config'])

        # train model on corpus
        _, model = apply_text_model(model, corpus, mode='fit', return_model=True, **kwargs)

    # apply model to text
    embedded_text = apply_text_model(model, text, mode='transform', return_model=False, **kwargs)

    # turn array into dataframe
    df = wrangle_array(embedded_text, **array_args)

    if return_model:
        return df, model
    else:
        return df
