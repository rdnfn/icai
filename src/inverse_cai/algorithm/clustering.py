import random
import copy
from sklearn.cluster import KMeans
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

import inverse_cai.models
from inverse_cai.experiment.config import ExpConfig
import inverse_cai.algorithm.utils


def cluster_principles(
    principles: list,
    num_clusters: int,
    model_name: str,
    random_clusters: bool,
):
    """
    Cluster principles.
    """
    if random_clusters:
        return cluster_principles_random(principles, num_clusters)
    elif num_clusters >= len(set(principles)):
        logger.warning(
            (
                "Number of clusters is less than the number of unique principles. "
                "Clustering will be skipped, only direct string matches will be deduplicated."
            )
        )
        return {i: [principle] for i, principle in enumerate(set(principles))}
    else:
        return cluster_principles_with_embedding(principles, num_clusters, model_name)


def cluster_principles_random(
    principles: list,
    num_clusters: int,
):
    """
    Cluster principles randomly.
    """
    logger.info(f"Clustering principles randomly into {num_clusters} clusters")
    principles = copy.deepcopy(principles)
    principles_by_cluster = {}
    # shuffle the principles
    random.shuffle(principles)
    for i, principle in enumerate(principles):
        cluster = i % num_clusters
        if cluster not in principles_by_cluster:
            principles_by_cluster[cluster] = []
        principles_by_cluster[cluster].append(principle)

    return principles_by_cluster


def cluster_principles_with_embedding(
    principles: list,
    num_clusters: int,
    model_name: str,
):
    """
    Cluster principles.
    """

    # generating embedding for each principle
    logger.info("Generating embeddings for principles")
    embeddings_model = inverse_cai.models.get_embeddings_model(model_name)
    embeddings = embeddings_model.embed_documents(principles)
    logger.info("Embeddings generated")

    # clustering the embeddings
    logger.info(f"Clustering the embeddings using KMeans (clusters={num_clusters})")
    estimator = KMeans(n_clusters=num_clusters).fit(embeddings)
    labels = estimator.labels_
    logger.info("Clustering complete")

    principles_by_cluster = {}
    for i in range(num_clusters):
        principles_by_cluster[i] = []
        for j, label in enumerate(labels):
            if label == i:
                principles_by_cluster[i].append(principles[j])

    return principles_by_cluster


def print_clusters(principles_by_cluster, summaries=None):
    """
    Print clusters.
    """
    for i, cluster in principles_by_cluster.items():
        if not summaries:
            print(f"##### Cluster {i}")
        else:
            print(f"##### Cluster {i}: {summaries[i]}")
        for i, principle in enumerate(cluster):
            print(f"  {i+1}. {principle}")
        print()


def get_cluster_summaries(
    principles_by_cluster,
    model_name,
    sample_instead_of_rewrite,
    config: ExpConfig,
    prompt_principles: bool = False,
):
    """
    Get summaries for each cluster.

    Either by sampling one of the principles, or by summarizing all
    principles in the cluster into a new single principles using a language model.
    """
    summaries = {}
    for i, principles in principles_by_cluster.items():
        if len(principles) == 1:
            summaries[i] = principles[0]
        elif sample_instead_of_rewrite:
            summaries[i] = random.choice(principles)
        else:
            summaries[i] = summarize_cluster(
                principles,
                model_name=model_name,
                config=config,
                prompt_principles=prompt_principles,
            )
    return summaries


def summarize_cluster(
    single_cluster_principles,
    model_name,
    config: ExpConfig,
    prompt_principles: bool = False,
):
    """
    Given a cluster of principles, summarize the cluster.
    """

    messages = inverse_cai.algorithm.utils.parse_prompt(
        prompt_str=(
            config.alg_prompts.prompt_cluster_summary_prompt
            if prompt_principles
            else config.alg_prompts.cluster_summary_prompt
        ),
        prompt_kwargs=dict(
            principles="\n\n".join(single_cluster_principles),
        ),
    )

    model = inverse_cai.models.get_model(model_name)

    summary = model.invoke(messages).content
    return summary
