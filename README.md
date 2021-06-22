# metadl_competition_fact_sheet
Repository for generating Fact Sheet for MetaDL Competition

## Step 1 - Generate Fact Sheet Results

`generate_fact_sheet_results.py` `generate_fact_sheet_results.ipynb`


## Step 3
`generate_html_report.py`

Use the following command to create a PDF report from the results generated in **Step 1**

```
python generate_html_report.py \
--results_dir "./experiment_1_results" \
--title "Fact Sheet Experiment # 1"
```

## template.html