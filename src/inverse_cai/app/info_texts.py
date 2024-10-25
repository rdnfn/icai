"""Module for creating information texts in app."""

METHOD_INFO_HEADING = "🔬 Method details"

METHOD_INFO_TEXT = """
**TLDR.** *Inverse Constitutional AI* (ICAI) helps understand the implicit goals encoded in pairwise feedback data.

**Background.**
Many ML researchers use pairwise feedback as an optimization objective, both for training or evaluation. However, few really understand *what implicit goals* they are training (or evaluating) their models towards, other than "aligning with human values". ICAI helps to understand these implicit goals by compressing large amounts of pairwise feedback data into individual principles, each a possible explanation of annotators' decisions.

**Method.**
Our initial *Inverse Constitutional AI* (ICAI) implementation consists of two core steps: (A) *proposing hypothesis principles* that *might* explain annotators' decisions, and (B) *evaluating each principle* by how well an LLM following it is able to reconstruct the original pairwise feedback data. Below you can see the principles with the corresponding reconstruction performance metrics. See the [paper](https://arxiv.org/abs/2406.06560) for full details on the ICAI method.

**Interpretation of results.**
A principle that performs (i.e. reconstructs the feedback) well might be a good explanation of the annotators' reasoning process – or at least correlates with that process. Using the corresponding feedback for training may lead to models that follow these principles. For example, if the principle *"select the more concise response"* reconstructs a feedback dataset well, a model trained on this dataset may become more concise. Similarly, in the context of evaluation, if such a principle shows up in feedback that prefers a certain model, then this model likely follows this principle.
"""
