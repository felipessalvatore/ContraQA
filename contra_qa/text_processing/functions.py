import re

spaces = re.compile(' +')


def remove_first_space(x):
    """
    :param x: word
    :type x: str
    :return: word withou space in front
    :rtype: str
    """
    if x[0] == " ":
        return x[1:]
    else:
        return x


def simple_pre_process_text_df(data):
    """
    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    """

    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x))) # noqa
    data['text'] = data['text'].apply(remove_first_space) # noqa remove space in the first position
    data['text'] = data['text'].apply((lambda x: spaces.sub(" ", x))) # noqa remove double spaces
