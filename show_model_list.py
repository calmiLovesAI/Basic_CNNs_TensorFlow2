from models import get_model2idx_dict

if __name__ == '__main__':
    print("Models included: ")
    for model_name, i in get_model2idx_dict().items():
        print("MODEL: {}, INDEX: {}".format(model_name, i))
