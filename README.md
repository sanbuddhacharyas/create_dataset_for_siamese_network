1)creating dataset:
    copy dataset folder in the main directory "create_dataset_for_siamese_network"
    hierarcy:
        dataset/
            items/
                .png

    use flush.py cell to start from starting
    Donot use  flush.py to resume from last step
    
    For  create dataset: Note: "a" ==> "Back" , "s"==>similar , "d" ==>disimilar, "q"==>quit

2)rechecking dataset:
    copy dataset and dataset_sim.txt inside a folder "your folder name"
    change the parameter dataset_name="your folder name" in the recheck_created_dataset.ipynb
    For rechecking if the dataset recheck_created_dataset: "a"==>"Back", "s"==>switch(similar to dissimilar or dissimilar to similar), "d"==>forward, "q"==>quit