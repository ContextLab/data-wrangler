import six
import os
import warnings
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
    :param x: the object to test

    Returns
    -------
    :return: True if x contains "transform", "fit", and "fit_transform" methods and False otherwise.
    """
    return hasattr(x, 'transform') and hasattr(x, 'fit') and hasattr(x, 'fit_transform')


def is_hugging_face_model(x):
    """
    Determine whether an object seems to be a valid hugging-face model

    Parameters
    ----------
    :param x: the object to test

    Returns
    -------
    :return: True if x contains an "embed" method, and False otherwise.
    """
    return hasattr(x, 'embed')


def robust_is_sklearn_model(x):
    """
    Wrapper for is_sklearn_model that also supports strings-- e.g., the string 'SparsePCA' will be a valid scikit-learn
    model when checked with this function, because 'SparsePCA' is defined in the sklearn.decomposition module.

    Parameters
    ----------
    :param x: a to-be-tested model object or a string

    Returns
    -------
    :return: True if x (or the scikit-learn module x evaluates to) contains "transform", "fit", and "fit_transform"
      methods and False otherwise.
    """
    x = get_text_model(x)
    return is_sklearn_model(x)


def robust_is_hugging_face_model(x):
    """
    Wrapper for is_hugging_face_model that also supports strings-- e.g., the string 'WordEmbeddings' will be a valid
    hugging-face model when checked with this function, because 'WordEmbeddings' is defined in the flair.embeddings
    module and contains an "embed" method.
    ----------
    :param x: a to-be-tested model object or a string

    Returns
    -------
    :return: True if x (or the hugging-face module x evaluates to) contains an "embed" method and False otherwise.
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
    :param x: an object to turn into a valid scikit-learn or hugging-face model (e.g., an already-valid model or a
      string)

    Returns
    -------
    :return: A valid scikit-learn or hugging-face model (or None if no model matching the given description can be
      found)
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

    [Parameters]
    ----------
    :param dataset_name: a string containing the corpus name.  Can be one of the following:
      - Corpora built into data-wrangler:
        - 'minipedia': a curated and cleaned up subset of Wikipedia containing articles on a wide variety of topics
        - 'neurips': a collection of NeurIPS articles
        - 'sotus': transcripts of state of the union addresses from US Presidents from 1989 -- 2018
        - 'khan': transcripts of (most) Khan Academy YouTube videos
      - Any hugging-face corpus; for a full list see https://huggingface.co/datasets
        Note that downloading hugging-face corpora also requires specifying a config_name
    :param config_name: configuration name or description for hugging-face corpora.  This argument is ignored if dataset
      name is set to one of the data-wrangler corpora described above.

    Returns
    -------
    :return: A list of number-of-documents strings, where each string contains the text of one document in the corpus.
    """

    key = f'{dataset_name}[{config_name}]'
    if key in preloaded_corpora.keys():
        return preloaded_corpora[key]

    def get_formatter(s):
        return s[s.find('_'):(s.rfind('_') + 1)]

    # built-in corpora
    corpora = {
        'minipedia': 'https://www.dropbox.com/s/eal65nd5a193pmk/minipedia.npz?dl=1',
        'neurips': 'https://www.dropbox.com/s/i32dycxr0qa90wx/neurips.npz?dl=1',
        'sotus': 'https://www.dropbox.com/s/e2qfw8tkmxp6bad/sotus.npz?dl=1',
        'khan': 'https://www.dropbox.com/s/ieztnyhao2ejo48/khan.npz?dl=1'}

    if dataset_name in corpora.keys():
        print(f'loading corpus: {dataset_name}', end='')
        data = load(corpora[dataset_name], dtype='numpy')
        try:
            corpus = data['corpus']
            print('...done!', end='')
            preloaded_corpora[key] = corpus
            return corpus
        finally:
            # ensure NpzFile is closed
            data.close()
        print('')

    # Hugging-Face Corpus
    try:
        data = load_dataset(dataset_name, config_name)
    except FileNotFoundError:
        raise RuntimeError(f'Corpus not found: {dataset_name}.  Available corpora: {", ".join(list_datasets())}')
    except ValueError:
        raise RuntimeError(f'Configuration for {dataset_name} corpus not found: {config_name}. '
                           f'Available configurations: {", ".join(get_dataset_config_names(dataset_name))}')

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
    """
    Apply a scikit-learn or hugging-face text embedding model to one or more text datasets.  Scikit-learn models are
    trained on the specified corpus and then applied to all datasets.  All Hugging-Face models are pre-trained.

    Parameters
    ----------
    :param x: the model to apply.  Supported models include:
      - Scikit-learn models.  The recommended pipeline is to specify a feature extraction model (for turning text into
        a number-of-documents by number-of-features matrix), and then to apply a matrix decomposition or embedding model
        (for turning the features matrix into text embeddings).  When models are passed as a list, each model is applied
        in succession to the output of the previous model.  The pipeline is first fit to the provided corpus, and then
        applied to the given text.  Default: ['CountVectorizer', 'LatentDirichletAllocation']
        - All scikit-learn text feature extraction models are supported; for a full list see
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
            These may be passed either as callable modules (e.g., sklearn.feature_extraction.text.CountVectorizer) or
            as strings (e.g., 'CountVectorizer').  Default options for each model are defined in config.ini.
        - All scikit-learn matrix decomposition models are supported; for a full list see
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
            These may be passed either as callable modules (e.g., sklearn.decomposition.NMF) or as strings (e.g.,
            'NMF').  Default options for each model are defined in config.ini.
      - Hugging-face models.  These take raw text as input and produce text embeddings as output.  Hugging-face models
          are specified using dictionaries containing the following keys:
            - 'model': the type of embedding to use-- any flair embedding type is supported; for a full list see
                https://github.com/flairNLP/flair#tutorials.  For word-level embeddings, 'WordEmbeddings' is
                recommended.  For document-level embeddings, we recommend using either 'TransformerDocumentEmbeddings'
                (to model the full document's content) or 'SentenceTransformerDocumentEmbeddings' (if sentence-level
                representations are needed).  Embeddings may be specified either as a string (e.g.,
                'TransformerDocumentEmbeddings') or as a callable module (e.g.,
                flair.embeddings.TransformerDocumentEmbeddings).
            - 'args': a list of unnamed arguments to pass to the given model.  All pre-trained hugging-face models are
                supported, including (but not limited to):
                  - word-level embedding models:
                      https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md
                  - sentence-level transformer models:
                      https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
                  - document-level transformer models: https://huggingface.co/transformers/pretrained_models.html
            - 'kwargs': a dictionary of keyword arguments to pass to the given model (these are model-specific; for
                 details and examples see https://github.com/flairNLP/flair#tutorials
          for example, to embed a document using GPT-2, use
            {model: 'TransformerDocumentEmbeddings', args: ['gpt2'], 'kwargs': {}}
          The 'kwargs' dictionary may be further subdivided; if an 'embedding_kwargs' key is included in 'kwargs',
          its values will be treated as keyword arguments to be applied to the embedding model when it is initialized.
          All other keyword arguments are passed on to flair.data.Sentence in order to tokenize the given text.
    :param text: a string (a single word, sentence, or document), list of strings (a list of words, sentences, or
      documents), or a nested list of strings (a list of listed words, sentences, or documents).  Strings and (shallow)
      lists of strings result in a single embedding matrix; nested lists produce a list of embedding matrices (one
      per lowest-level list)
    :param args: a list of unnamed arguments to pass to *every* text embedding model or pipeline step.  Default: [].
    :param mode: one of: 'fit' (fit the model), 'transform' (apply an already-fitted model), or 'fit_transform' (fit
      a model and then apply it to the same text).  The 'fit' mode is only supported for scikit-learn (and scikit-learn-
      compatible) models.
    :param return_model: if True, return both the embedded text and a trained model that may be applied to new text. If
      False, return only the text embeddings.  Default: False.
    :param kwargs: keyword arguments are passed to the embedding model; these are equivalent to specifying the
      embedding model as a dictionary.  When a keyword argument appears in both model['kwargs'] and kwargs, the kwargs
      value is used preferentially.

    Returns
    -------
    :return: The text embeddings (if return_model is False) or a tuple whose first element is the text embeddings and
      whose second element is a fitted model that may be applied to new text (if return_model is True).
    """
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
        warnings.simplefilter('ignore')

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
    """
    Parse, load, or download one or more documents.

    Parameters
    ----------
    :param x: A string or list of strings.  Each string can be either the text of a document, a file path, or a URL.  If
      a file path or URL is provided, the contents are loaded in, treated as text, and returned.  If a list of strings
      is provided, the get_text function is applied to each element of the list.
    :param force_literal: If True, interpret strings literally (rather than checking to see if the strings point to a
      local or remote file).  Default: False.

    Returns
    -------
    :return: The text as a string or (potentially nested) list of strings
    """
    if type(x) == list:
        return [get_text(t) for t in x]
    if (type(x) in six.string_types) or (type(x) == np.str_):
        if os.path.exists(x):
            if not force_literal:
                return get_text(load(x), force_literal=True)
        return x
    return None


def is_text(x):
    """
    Test whether an object contains (or points to) text.

    Parameters
    ----------
    :param x: the object to test

    Returns
    -------
    :return: True if the object is (or points to) text and False otherwise.
    """

    if type(x) == list:
        return all([is_text(i) for i in x])
    return get_text(x) is not None


def to_str_list(x, encoding='utf-8'):
    """
    Internal helper function used to wrangle text data.  Handles binary strings, nested lists of strings, and arrays
      or dataframes containing text.

    Parameters
    ----------
    :param x: the text-containing object to be wrangled.
    :param encoding: for objects of type bytes, specify the encoding.  Default: 'utf-8'.

    Returns
    -------
    :return: a string or (possibly nested) list of strings
    """
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
    """
    Turn text into DataFrames

    Parameters
    ----------
    :param text: A string or (nested) list of strings.  Each string can contain either the to-be-wrangled text, a file
      path, or a URL.
    :param return_model: if True, return a fitted model that may be applied to new text data, along with the wrangled
      text.  Default: False.
    :param kwargs: Other (optional) keyword arguments may be passed into the function to control the wrangling
      process:
      - 'corpus': any built-in or hugging-face corpus (see get_corpus for more details); this argument is passed to the
        get_corpus function as the "dataset_name" keyword argument
        - the 'config' argument may be used to select a specific variant of the corpus (passed to get_corpus as the
          "config_name" keyword argument).
      - 'model': any scikit-learn-compatible or hugging-face-compatible model (see apply_text_model for more details)
      - 'array_kwargs': a dictionary of keyword arguments that may be passed to wrangle_array to control how the final
        DataFrame is structured (see wrangle_array for details).

    Returns
    -------
    :return: a DataFrame (or list of DataFrames) containing the embedded text.  If return_model is True a tuple, whose
      first element contains the embedded text and second element contains the fitted models, is returned instead.
    """
    text = get_text(text)
    if type(text) is not list:
        text = [text]

    model = kwargs.pop('model', eval(defaults['text']['model']))
    corpus = kwargs.pop('corpus', None)
    config = kwargs.pop('config', None)
    array_kwargs = kwargs.pop('array_kwargs', {})

    if type(model) is not list:
        model = [model]

    if any(robust_is_sklearn_model(m) for m in model):
        if corpus is not None:
            if not ((type(corpus) is list) and is_text(corpus)):
                corpus = get_corpus(dataset_name=corpus, config_name=config)
        else:
            corpus = get_corpus(dataset_name=eval(defaults['text']['corpus']),
                                config_name=eval(defaults['text']['corpus_config']))

        # train model on corpus
        _, model = apply_text_model(model, corpus, mode='fit', return_model=True, **kwargs)

    # apply model to text
    embedded_text = apply_text_model(model, text, mode='transform', return_model=False, **kwargs)

    # turn array into dataframe
    df = wrangle_array(embedded_text, **array_kwargs)

    if return_model:
        return df, model
    else:
        return df
