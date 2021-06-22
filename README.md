# metadl_competition_fact_sheet
Repository for generating Fact Sheet for MetaDL Competition

## Step 1 - Generate Fact Sheet Results

`generate_fact_sheet_results.py` `generate_fact_sheet_results.ipynb`

```
!python generate_factsheet_results.py \
--CSV_PATH '../Internship_Files/super-categories-and-categories.csv' \
--IMAGE_PATH '../Internship_Images/processed_images_128' \
--PREDICTIONS_PATH './Baseline_Super_Categories_Predictions_40_random' \
--LABEL_COLUMN 'Category' \
--IMAGE_COLUMN 'file_name' \
--CATEGORIES_TO_COMBINE 5 \
--IMAGES_PER_CATEGORY 10
```


## Step 3
`generate_html_report.py`

Use the following command to create a PDF report from the results generated in **Step 1**

```
python generate_pdf_report.py \
--results_dir "./experiment_1_results" \
--title "Fact Sheet Experiment # 1"
```

## template.html
