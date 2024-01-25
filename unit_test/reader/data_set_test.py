from kis.reader.data_set import KisDataSet


def data_set_test():
    data_dir = "../data/json/"
    train_file = data_dir + "dataset_1.json"
    test_file = data_dir + "dataset_2.json"

    for streaming_value in (True, False):
        ds_from_local = KisDataSet()
        ds_from_local.streaming = streaming_value

        ds_from_local.split = "train"
        ds_from_local.load_local_json_files(data_dir=data_dir, split_iterated='train')
        print("dataset from directory:")
        print(ds_from_local.dataset)
        for _ in range(20):
            sample = ds_from_local._next_sample()
            print(_, sample)
        ds_from_local.split = None

        ds_from_local.load_local_json_files(data_files={"train": train_file, "test": test_file})
        print("dataset from files:")
        print(ds_from_local.dataset)
        ds_from_local._set_split_iterator("test")
        for _ in range(15):
            sample = ds_from_local._next_sample()
            print(_, sample)
        ds_from_local._set_split_iterator("train")
        for _ in range(15):
            sample = ds_from_local._next_sample()
            print(_, sample)

        for _ in ds_from_local.split_to_generator("train", batch_num=2):
            print(len(_))
        for _ in ds_from_local.split_to_generator("train", batch_num=3):
            print(_)

        ds_from_local._batch_size = 3
        ds_from_local.load_local_json_files(data_files={"train": train_file, "test": test_file})
        batch_num = 0
        for _ in ds_from_local.split_to_generator("train"):
            batch_num += 1
        print(f"Number of batch in split 'train': {batch_num}")
        ds_from_local._batch_size = 4


if __name__ == "__main__":
    data_set_test()
