from datetime import datetime


def datetime_format_transfer(df, columns_list, format_str='%Y-%m-%dT%H:%M:%S.%fZ'):
    for column in columns_list:
        df[column] = df[column].apply(lambda x: datetime.strptime(x, format_str))
    return df
