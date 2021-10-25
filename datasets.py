
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from dataloaderdir.CUB import *
from dataloaderdir.AIRCRAFTloader import *
from dataloaderdir.Quickdraw import *
from dataloaderdir.Logo import *


def dataset(args, datanames):
    #MiniImagenet   
    train_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.train_shot,
                                      num_test_per_class=args.num_query)

    valid_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.test_shot,
                                      num_test_per_class=args.num_query)

    transform = Compose([Resize(84), ToTensor()])
    MiniImagenet_train_dataset = MiniImagenet(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=train_transform,
                                      download=True)

    Imagenet_train_loader = BatchMetaDataLoader(MiniImagenet_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    MiniImagenet_val_dataset = MiniImagenet(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=valid_transform)

    Imagenet_valid_loader = BatchMetaDataLoader(MiniImagenet_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    MiniImagenet_test_dataset = MiniImagenet(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=valid_transform)

    Imagenet_test_loader = BatchMetaDataLoader(MiniImagenet_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    #CUB dataset
    
    #transform = Compose([ToTensor()])
    transform = None
    #transform = Compose([Resize(84), ToTensor()])
    CUB_train_dataset = CUBdata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=train_transform,
                                      download=True)

    CUB_train_loader = BatchMetaDataLoader(CUB_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CUB_val_dataset = CUBdata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=valid_transform)

    CUB_valid_loader = BatchMetaDataLoader(CUB_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CUB_test_dataset = CUBdata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=valid_transform)
    CUB_test_loader = BatchMetaDataLoader(CUB_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    #Omniglot
    class_augmentations = [Rotation([90, 180, 270])]
    transform = Compose([Resize(84), ToTensor()])
    Omniglot_train_dataset = Omniglot(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=train_transform,
                                      download=True)

    Omniglot_train_loader = BatchMetaDataLoader(Omniglot_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Omniglot_val_dataset = Omniglot(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=valid_transform)

    Omniglot_valid_loader = BatchMetaDataLoader(Omniglot_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Omniglot_test_dataset = Omniglot(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=valid_transform)
    Omniglot_test_loader = BatchMetaDataLoader(Omniglot_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    transform = None
    Aircraft_train_dataset = Aircraftdata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=train_transform,
                                      download=False)


    Aircraft_train_loader = BatchMetaDataLoader(Aircraft_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Aircraft_val_dataset = Aircraftdata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=valid_transform)

    Aircraft_valid_loader = BatchMetaDataLoader(Aircraft_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Aircraft_test_dataset = Aircraftdata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=valid_transform)
    Aircraft_test_loader = BatchMetaDataLoader(Aircraft_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)





    transform = None
    Quickdraw_train_dataset = Quickdraw(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=train_transform,
                                      download=False)


    Quickdraw_train_loader = BatchMetaDataLoader(Quickdraw_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Quickdraw_val_dataset = Quickdraw(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=valid_transform)

    Quickdraw_valid_loader = BatchMetaDataLoader(Quickdraw_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Quickdraw_test_dataset = Quickdraw(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=valid_transform)
    Quickdraw_test_loader = BatchMetaDataLoader(Quickdraw_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)







    transform = None
    Necessities_folder = 'Necessities'
    Necessities_train_dataset = Logo(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=train_transform,
                                      download=False,
                                      folder = Necessities_folder)


    Necessities_train_loader = BatchMetaDataLoader(Necessities_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Necessities_val_dataset = Logo(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=valid_transform,
                                    folder = Necessities_folder)

    Necessities_valid_loader = BatchMetaDataLoader(Necessities_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Necessities_test_dataset = Logo(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=valid_transform,
                                     folder = Necessities_folder)
    Necessities_test_loader = BatchMetaDataLoader(Necessities_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


 
    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []
    for name in  datanames:
        if name == 'MiniImagenet':
            train_loader_list.append({name: Imagenet_train_loader})
            valid_loader_list.append({name: Imagenet_valid_loader})
            test_loader_list.append({name:  Imagenet_test_loader})

        if name == 'CUB':
            train_loader_list.append({name:CUB_train_loader})
            valid_loader_list.append({name:CUB_valid_loader})
            test_loader_list.append({name:CUB_test_loader})
        if name == 'Aircraft':
            train_loader_list.append({name:Aircraft_train_loader})
            valid_loader_list.append({name:Aircraft_valid_loader})
            test_loader_list.append({name:Aircraft_test_loader})
        if name == 'Omniglot':
            train_loader_list.append({name:Omniglot_train_loader})
            valid_loader_list.append({name:Omniglot_valid_loader})
            test_loader_list.append({name:Omniglot_test_loader})

        if name == 'Quickdraw':
            train_loader_list.append({name:Quickdraw_train_loader})
            valid_loader_list.append({name:Quickdraw_valid_loader})
            test_loader_list.append({name:Quickdraw_test_loader})

        if name == 'Necessities':
            train_loader_list.append({name:Necessities_train_loader})
            valid_loader_list.append({name:Necessities_valid_loader})
            test_loader_list.append({name:Necessities_test_loader})


    return  train_loader_list, valid_loader_list, test_loader_list 

