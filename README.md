# metadl_competition_fact_sheet
Repository for generating factsheet from dataset exploration for MetaDL Competition



## Step 1 - Generate Fact Sheet Results

To generate factsheet results, you can use either `generate_fact_sheet_results.ipynb` or `generate_fact_sheet_results.py` 


#### Ipython Notebook 
(skip this if you want to use python script)

In the notebook file `generate_fact_sheet_results.ipynb`, you have to set some variable in the first cell under the heading **Settings** and then run the whole notebook to the end.

- `CSV_PATH` : Path of the CSV file which contains the label and image name
- `CSV_WITH_TAB` : True if CSV is tab separated otherwise false
- `IMAGE_PATH` : Path of the directory where images to be used in this experiement are saved
- `PREDICTIONS_PATH`: Path of the directory where the results of this experiments will be saved logs.txt, super_categories.txt and one directory for each super_category which contains categories.txt, categories_auc.txt, logs.txt, some figures etc 
- `LABEL_COLUMN` : label column name in csv
- `IMAGE_COLUMN` : image column name in csv
- `CATEGORIES_TO_COMBINE` : number of categories to combine to make a super-category or a classification task
- `IMAGES_PER_CATEGORY` : number of images per category
- `MAX_EPISODES` : maximum limit on episodes/super-categories
- `SEED` : seed for generating super-categories by the same random combination of categories (Do not change)


#### Python Script
(skip this if you want to use ipython notebook)

You can use the script `generate_fact_sheet_results.py` to generate the factsheet results by execuring the following shell command with the required argumenets
 
```
python generate_factsheet_results.py \
--CSV_PATH './data.csv' \
--IMAGE_PATH './images' \
--PREDICTIONS_PATH './experiment_1_results' \
--LABEL_COLUMN 'Category' \
--IMAGE_COLUMN 'file_name' \
--CATEGORIES_TO_COMBINE 5 \
--IMAGES_PER_CATEGORY 10
```

***Optional Arguments***  
`--CSV_WITH_TAB` (default: False)  
`--MAX_EPISODES` (default: None)  



#### Results
The results generated in this step contains the following files in the ***PREDICTIONS*** directory:

    - logs.txt
    - super_categories.txt
    - one folder for each super_category (If you have 10 randomly generated super_categories, then you will see 10 folders named from 0-9)
    - One super_category folder contains the following files:
        - logs.txt
        - categories.txt
        - categores_auc.txt
        - train.csv
        - valid.csv
        - train_results.png
        - confusion_matrix.png
        - auc.png
        - auc_histogram.png
        - roc_curves.png
        - sample_images.png
        - wrongly_classified_images.png






## Step 2

Make sure to install **jinja2** and **pdfkit** before executing this step

Once you have your results from **Step 1**, you can now execute the python script `generate_pdf_report.py`.

Use the following command to create a PDF report from the generated results in the step above.

```
python generate_pdf_report.py \
--results_dir "./experiment_1_results" \
--title "Fact Sheet Experiment # 1"
```

This script will generate a pdf report using the html template `template.html`. The pdf file will be stored in a newly created directory with the name ***report_files***.

The PDF will have a summary of the results in a table and then individual results of super-categories/classification tasks.




## Note:
The categories/classes are combined in a way in Step 1 that no category is repeated in super-categories.  
