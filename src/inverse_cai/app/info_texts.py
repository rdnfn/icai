"""Module for creating information texts in app."""

METHOD_INFO_HEADING = "🔬 Method details"

METHOD_INFO_TEXT = """
**Background.**
Many ML researchers use pairwise feedback as an optimization objective, both for training or evaluation. However, few really understand *what implicit goals* they are training (or evaluating) their models towards, other than "aligning with human values". ICAI helps to understand these implicit goals by compressing large amounts of pairwise feedback data into individual principles, each a possible explanation of annotators' decisions.

**Method.**
Our initial *Inverse Constitutional AI* (ICAI) implementation consists of two core steps: (A) *proposing hypothesis principles* that *might* explain annotators' decisions, and (B) *evaluating each principle* by how well an LLM following it is able to reconstruct the original pairwise feedback data. Below you can see the principles with the corresponding reconstruction performance metrics. See the [paper](https://arxiv.org/abs/2406.06560) or the [code](https://github.com/rdnfn/icai) for full details on the ICAI method.

**Interpretation of results.**
A principle that performs (i.e. reconstructs the feedback) well might be a good explanation of the annotators' reasoning process – or at least correlates with that process. Using the corresponding feedback for training may lead to models that follow these principles. For example, if the principle *"select the more concise response"* reconstructs a feedback dataset well, a model trained on this dataset may become more concise. Similarly, in the context of evaluation, if such a principle shows up in feedback that prefers a certain model, then this model likely follows this principle.

**Sources of variability.**
There are many different reasons why one set of annotator or annotations leads to different rules reconstructing it well:
1. *Contradicting annotator preferences:* the underlying annotator preferences are truly different and contradicting, and thus principles achieve different performance on each dataset.
2. *Different (possibly non-contradicting) revealed preferences:* in many datasets (e.g. PRISM, Chatbot Arena) the annotators choose themselves what to talk about with the AI. Whilst such annotators may overall agree (e.g., they would label each other's interactions similarly), revealed preference may still reveal something about what principles are more important to them for their use-cases --- even if their preferences are not contradictory.
3. *Noisy LLM reconstructions:* Some variability may also be explained by the underlying LLM not consistently using one rule in a certain way. Whilst some principles may be clearer to interpret, others can be more vague (and thus lead to noisy reconstructions). Running ICAI on larger datasets should mitigate this effect to certain degree.

"""

TLDR_TEXT = """
**TLDR:** An app to understand what principles pairwise feedback is teaching or testing in our models, e.g. is the pairwise feedback asking for more or less concise output?

**What is this app?** The *Inverse Constitional AI* (ICAI) App helps interpret pairwise feedback datasets. The app helps identify *principles* that annotators may have (implictly) followed to provide pairwise feedback, e.g. *"select the more concise response"*. Each principle is tested by measuring the ability of an AI model prompted to follow the principle to (blindly) reconstruct the original feedback preferences. A principle performing well in this reconstruction task will possibly transfer to downstream use-cases. Note that the app shows results from the ICAI algorithm, but does not currently run the algorithm live.


**How can I intrepret the results?** For example, assume our pairwise feedback data has a well-performing principle *"select the more concise response"*. When used for training, such a dataset may teach a model to *make more concise responses*. When used for *evaluation* (e.g. like [Chatbot Arena](https://lmarena.ai/)), such a data will likely *rank more concise models higher*.



**What is pairwise feedback?** Pairwise feedback typically consists of a prompt, two corresponding AI model responses, and an annotation picking the better response. Such feedback is very widely used for state-of-the-art AI models, both to train (e.g. [RLHF](https://arxiv.org/abs/2312.14925)) and evaluate (e.g. [Chatbot Arena](https://lmarena.ai/)). Yet, often such datasets are used as a "black-box oracle", without a good understanding of what properties they teach or test in our models.

"""
