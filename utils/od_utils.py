def if_in_bbox(transactions):
    transactions['in_bbox'] = ((113.764635 < transactions['destination_log'])
                               & (transactions['destination_log'] < 114.608972)
                               & (22.454727 < transactions['destination_lat'])
                               & (transactions['destination_lat'] < 22.842654)
                               & (113.764635 < transactions['original_log'])
                               & (transactions['original_log'] < 114.608972)
                               & (22.454727 < transactions['original_lat'])
                               & (transactions['original_lat'] < 22.842654))
    return transactions