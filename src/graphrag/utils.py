import pandas as pd


def add_ids_to_edges(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add source_id and target_id to the edges DataFrame based on the nodes DataFrame.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node information with 'id' and 'name' columns.
        edges_df (pd.DataFrame): DataFrame containing edge information with 'source' and 'target' columns.

    Returns:
        pd.DataFrame: Updated edges DataFrame with 'source_id' and 'target_id' columns added.
    """
    # Get columns of interest
    edge_cols = ['source', 'target', 'weight', 'description', 'source_degree', 'target_degree', 'rank']
    edges_df = edges_df[edge_cols]
    # Merge for source_id
    updated_edges = edges_df.merge(nodes_df[['id', 'name']], left_on='source', right_on='name', how='left')
    updated_edges = updated_edges.rename(columns={'id': 'source_id'}).drop('name', axis=1)

    # Merge for target_id
    updated_edges = updated_edges.merge(nodes_df[['id', 'name']], left_on='target', right_on='name', how='left')
    updated_edges = updated_edges.rename(columns={'id': 'target_id'}).drop('name', axis=1)

    # Reorder columns
    column_order = ['source', 'target', 'source_id', 'target_id'] + [
        col for col in updated_edges.columns if col not in ['source', 'target', 'source_id', 'target_id']
    ]
    updated_edges = updated_edges[column_order]

    return updated_edges