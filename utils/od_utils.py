def if_in_bbox(transactions):
    """
    Add a column indicating whether the geodetic coordinate of origin and destination for the transaction is within the
    bounding box.
    :param transactions: the DataFrame to be processed.
    :return:
    """
    transactions['in_bbox'] = ((113.764635 < transactions['destination_log'])
                               & (transactions['destination_log'] < 114.608972)
                               & (22.454727 < transactions['destination_lat'])
                               & (transactions['destination_lat'] < 22.842654)
                               & (113.764635 < transactions['original_log'])
                               & (transactions['original_log'] < 114.608972)
                               & (22.454727 < transactions['original_lat'])
                               & (transactions['original_lat'] < 22.842654))
    return transactions


def filter_in_bbox(transactions):
    """
    Return the transactions within the bounding box.
    :param transactions:
    :return:
    """
    if_in_bbox(transactions)
    transactions = transactions.loc[transactions["in_bbox"]].reset_index(drop=True)
    transactions.drop("in_bbox", axis=1, inplace=True)
    return transactions
