from load_data import instance_custom_training

model=instance_custom_training()
model.modelConfig( network_backbone="resnet101" ,num_classes=2,batch_size=8,class_names=['gia sat','gia go'])
model.load_pretrained_model()
model.load_dataset("data_gia")
model.visualize_sample()
model.train_model(num_epochs=30,path_trained_models="Models",layers="all")