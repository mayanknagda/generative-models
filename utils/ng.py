import torch
from typing import List, Dict, Any
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def get_ng_data(batch_size=128) -> Dict[str, Any]:
    """
    Used to get PyTorch dataloader for 20 Newsgroups dataset.
    Returns:
        train_dl: PyTorch dataloader for training data
        val_dl: PyTorch dataloader for validation data
        test_dl: PyTorch dataloader for test data
        vocab: Dictionary of words and their indices (word: idx)
    """
    train_text = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    val_text = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes")
    )
    test_text = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes")
    )
    vectorizer = CountVectorizer(min_df=20, max_df=0.90)
    train_data = vectorizer.fit_transform(train_text.data).toarray()
    val_data = vectorizer.transform(val_text.data).toarray()
    test_data = vectorizer.transform(test_text.data).toarray()
    vocab = vectorizer.vocabulary_
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float())
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float())

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return {"train_dl": train_dl, "val_dl": val_dl, "test_dl": test_dl, "vocab": vocab}


if __name__ == "__main__":
    data = get_ng_data()
    train_dl = data["train_dl"]
    vocab = data["vocab"]
    for batch in train_dl:
        print(batch[0].shape)
        break
    print(vocab)  # ('word': idx)
