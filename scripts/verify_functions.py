"""Comprehensive direct-evidence verification: run every major datawrangler function on real inputs
and print the actual outputs for examination. Run from the repo root."""
import numpy as np
import pandas as pd
import datawrangler as dw
from datawrangler.io.io import get_local_fname
from sklearn.feature_extraction.text import CountVectorizer

R = 'tests/resources'
def hdr(s): print("\n" + "=" * 4, s, "=" * 4)


print("datawrangler.__version__ =", dw.__version__)

# ============ ZOO: wrangle every supported datatype ============
hdr("ZOO / wrangle")
arr = dw.wrangle(np.array([[1, 2, 3], [4, 5, 6]]))
print("array  ->", type(arr).__name__, arr.shape, "| values:", arr.values.tolist())

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
print("dataframe ->", type(dw.wrangle(df)).__name__, "| preserved values:", dw.wrangle(df).values.tolist())

nul = dw.wrangle(None)
print("null   ->", type(nul).__name__, "| empty:", len(nul) == 0)

polars_df = dw.wrangle(np.arange(6).reshape(3, 2), backend='polars')
print("array (backend=polars) ->", type(polars_df).__name__, polars_df.shape)

img = dw.io.load(f'{R}/wrangler.jpg')
img_df = dw.wrangle(img)
print("image  ->", type(img_df).__name__, img_df.shape, "| pixel mean: %.2f" % img_df.values.mean())

# text via sklearn vectorization (CountVectorizer -> LDA) trained on the built-in 'minipedia' corpus
lda = dw.wrangle(['the cat sat on the mat', 'dogs run in the park', 'cats and dogs are pets'],
                 text_kwargs={'model': ['CountVectorizer', 'LatentDirichletAllocation'], 'corpus': 'minipedia'})
print("text (sklearn CountVectorizer->LDA) ->", type(lda).__name__, lda.shape)

# text via sentence-transformers embedding (pre-trained HF model)
emb = dw.wrangle(['hello world', 'data wrangler rocks'], text_kwargs={'model': 'all-MiniLM-L6-v2'})
print("text (ST embed all-MiniLM-L6-v2) ->", type(emb).__name__, emb.shape, "| mean: %.4f" % emb.values.mean())

# mixed list of datatypes, with dtype detection
mixed, dtypes = dw.wrangle([np.array([1, 2, 3]), 'some free text', df],
                           text_kwargs={'model': 'all-MiniLM-L6-v2'}, return_dtype=True)
print("mixed list -> detected dtypes:", dtypes, "| n results:", len(mixed))

# is_<type> predicates
print("is_array/is_dataframe/is_text/is_null:",
      dw.zoo.is_array(np.array([1, 2])), dw.zoo.is_dataframe(df), dw.zoo.is_text('hi'), dw.zoo.is_null([]))

# text helpers
print("to_str_list:", dw.zoo.to_str_list(['a', 'b']))
model = dw.zoo.get_text_model('all-MiniLM-L6-v2')
print("get_text_model('all-MiniLM-L6-v2') ->", type(model).__name__ if not isinstance(model, dict) else model)

# ============ DECORATE ============
hdr("DECORATE")


@dw.funnel
def colmeans(x):
    return x.mean(axis=0)


print("funnel(np.array) ->", colmeans(np.array([[1, 2], [3, 4]])).values.tolist())


@dw.decorate.list_generalizer
def square(x):
    return x ** 2


print("list_generalizer:", square(3), square([2, 3, 4]))

d1, d2 = df.iloc[:1], df.iloc[1:]
stacked = dw.stack([d1, d2])
print("stack -> shape", stacked.shape, "| is_multiindex:", dw.zoo.is_multiindex_dataframe(stacked))
print("unstack -> frame shapes:", [f.shape for f in dw.unstack(stacked)])


@dw.decorate.apply_unstacked
def per_frame_mean(x):
    return pd.DataFrame(x.mean(axis=0)).T


means = per_frame_mean(stacked)
print("apply_unstacked (per-frame means) -> is_multiindex:", dw.zoo.is_multiindex_dataframe(means))

imp = df.copy().astype(float)
imp.loc[0, 'a'] = np.nan


@dw.decorate.interpolate
def ident(x):
    return x


recovered = ident(imp, interp_kwargs={'impute_kwargs': {'model': 'IterativeImputer'}})
print("interpolate (impute IterativeImputer) -> no NaN:", not recovered.isna().any().any(),
      "| values:", recovered.values.round(3).tolist())

# ============ IO ============
hdr("IO")
print("load(csv, index_col=0) ->", dw.io.load(f'{R}/testdata.csv', index_col=0).shape)
print("load(txt) ->", repr(dw.io.load(f'{R}/home_on_the_range.txt')[:40]))
print("load_dataframe(csv) ->", dw.io.load_dataframe(f'{R}/testdata.csv').shape)
# save/load round-trip via the cache
key = 'verify://obj.pkl'
dw.io.save(key, {'x': [1, 2, 3]}, dtype='pickle')
print("save+load(pickle) round-trip ->", dw.io.load(get_local_fname(key), dtype='pickle'))
# remote load
try:
    remote = dw.io.load('https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/testdata.csv')
    print("load(remote URL) ->", type(remote).__name__, remote.shape)
except Exception as e:
    print("load(remote URL) -> skipped (network):", e)

# ============ UTIL ============
hdr("UTIL")
print("btwn in-range:", dw.util.btwn(np.array([1, 2, 3]), 0, 5),
      "| btwn out-of-range:", dw.util.btwn(np.array([1, 9]), 0, 5))
print("array_like([1,2,3]):", dw.util.array_like([1, 2, 3]),
      "| dataframe_like(df):", dw.util.dataframe_like(df))
print("depth: scalar", dw.util.depth(5), "| [1,2,3]", dw.util.depth([1, 2, 3]),
      "| [[1],[2]]", dw.util.depth([[1], [2]]))

# ============ CORE ============
hdr("CORE")
opts = dw.core.get_default_options()
print("get_default_options -> has CountVectorizer/text/data:",
      all(k in opts for k in ['CountVectorizer', 'text', 'data']))
cv = dw.core.apply_defaults(CountVectorizer)()
print("apply_defaults(CountVectorizer) -> stop_words:", cv.get_params()['stop_words'],
      "| max_df:", cv.get_params()['max_df'])
print("update_dict:", dw.core.update_dict({'a': 1, 'b': 2}, {'b': 9, 'c': 3}))
dw.core.set_dataframe_backend('polars')
print("set/get_dataframe_backend -> ", dw.core.get_dataframe_backend())
dw.core.reset_dataframe_backend()
print("reset_dataframe_backend -> ", dw.core.get_dataframe_backend())

# built-in corpus retrieval (small, cached)
hdr("CORPUS")
sotus = dw.zoo.text.get_corpus('sotus')
print("get_corpus('sotus') -> n docs:", len(sotus), "| first 40 chars:", repr(sotus[0][:40]))

print("\nALL MAJOR FUNCTIONS EXERCISED SUCCESSFULLY")
