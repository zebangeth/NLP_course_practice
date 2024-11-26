Final Project Report
Advancing Salary Transparency Through NLP: A Comparative Study of Two Approaches for Predicting Salaries from Job Descriptions

## 1. Introduction

### 1.1 Problem Statement and Motivation

### 1.2 Project Objectives and Scope

### 1.3 Overview of Approach

## 2. Data Description and Preprocessing

To verify the effectiveness of our models, I first generate synthetic datasets with known patterns and relationships to train and test the models. This allows us to validate our models' ability to capture salary-relevant information from unstructured text.

### 2.1 Synthetic Dataset

#### 2.1.1 Data Generation Process and Assumptions

**Data Generation Architecture**

The synthetic data generator is implemented as a Python class (`SyntheticJobDataGenerator`) that produces job postings with the following key components:
- Job titles and seniority levels
- Location information
- Industry classification
- Required skills and skill categories
- Detailed job descriptions
- Salary ranges (minimum, maximum, and median)

**Key Assumptions and Rules**

1. **Salary Base Ranges**
   The base salary ranges are stratified by seniority level:
   - Entry Level: $50,000 - $80,000
   - Mid Level: $80,000 - $120,000
   - Senior: $120,000 - $180,000
   - Lead: $150,000 - $220,000
   - Executive: $200,000 - $350,000

2. **Geographic Compensation Adjustment**
   Location-based salary multipliers reflect real-world cost of living variations:
   - San Francisco: 1.30x (highest)
   - New York: 1.20x
   - Seattle: 1.15x
   - Chicago: 1.10x
   - Austin: 1.00x (baseline)

3. **Industry Impact**
   Industry-specific multipliers model sector-based compensation differences:
   - Technology: 1.20x
   - Finance: 1.15x
   - Consulting: 1.10x
   - Healthcare: 1.00x
   - Manufacturing: 0.95x
   - Retail: 0.90x
   - Hospitality: 0.85x
   - Education: 0.80x

4. **Skill-Based Compensation**
   The generator implements a sophisticated skill-based compensation model:
   - Each additional skill contributes a 3% increase to the base salary
   - Skills are categorized into domains (e.g., Software Engineering, Data & AI, Product & Management)
   - Skill categories carry different weight multipliers:
     - Data & AI: 1.30x
     - Software Engineering: 1.25x
     - Finance: 1.20x
     - Product & Management: 1.15x
     - Design: 1.10x
     - Healthcare: 1.10x
     - Operations: 1.05x
     - Business: 1.00x

5. **Description Generation**
   Job descriptions are generated using templates that incorporate:
   - Seniority level and role
   - Required skills and their categories
   - Location and industry context
   - Compensation-relevant terminology

**Randomization and Variation**

To introduce realistic variation while maintaining the underlying patterns:
- A ±5% random variation is applied to final salary calculations
- Skills are selected with consideration for category relationships
- Description templates are randomly selected and populated
- Skills are sampled based on role-appropriate categories


#### 2.1.2 Validation of Synthetic Data

To validate the synthetic dataset's ability to serve as a reliable testing ground for our salary prediction models, I conducted an analysis of the generated data. The analysis focused on verifying that the data exhibits the intended patterns and relationships while maintaining realistic variations.

**Example Generated Job Posting Data Entry**:
```
Title: Mid Level Product Manager
Location: Seattle
Industry: Transportation
Skills: ['Retail Sales', 'Project Management', 'Business Development', 'Marketing Strategy', 'Agile Methodologies', 'Risk Management']
Description: Exciting opportunity for a Mid Level Product Manager in Seattle. Our Transportation division is expanding and needs someone skilled in Retail Sales, Project Management, Business Development, Marketing Strategy, Agile Methodologies and Risk Management. Competitive salary and comprehensive benefits package. Strong background in Product & Management and Business is essential.
Min Salary: $10,8746.65
Max Salary: $16,8703.84
```

**Dataset Overview**
- Total number of generated job postings: 10,000
- Med Salary range: $49,410 - $450,383
- Mean salary: $175,822

**Geographic Salary Distribution**

The analysis of median salaries by location confirms that our synthetic data accurately reflects the intended geographic compensation variations:

1. San Francisco: $195,906 (highest)
2. New York: $184,264
3. Seattle: $178,537
4. Chicago: $165,912
5. Austin: $154,208 (baseline)

**Industry Salary Distribution**

The industry-wise salary distribution shows clear segmentation that mirrors real-world compensation patterns:

1. Technology: $220,900
2. Finance: $201,530
3. Consulting: $195,907
4. Healthcare: $181,805
5. Real Estate: $170,952
6. Manufacturing: $168,826
7. Retail: $163,126
8. Transportation: $162,045
9. Hospitality: $151,011
10. Education: $140,498

The synthetic dataset exhibits several key properties that make it suitable for validating our salary prediction models:

1. **Controlled Relationships**:
   - Clear correlation between seniority and base salary
   - Consistent location-based adjustments
   - Industry-specific salary patterns
   - Skill category impact on compensation

2. **Realistic Complexity**:
   - Multiple interacting factors influence final salary
   - Related skills appear together (e.g., Python, SQL, AWS for software engineering roles)
   - Natural variation through random factors

3. **Text-Salary Alignment**:
   The generated job descriptions systematically encode salary-relevant information through:
   - Explicit mention of required skills
   - Industry context
   - Seniority level indicators
   - Location information

This synthetic dataset serves as a controlled environment for initial model validation, allowing me to verify whether our models can:
- Extract salary-relevant information from unstructured text
- Learn the underlying relationships between job attributes and compensation
- Generalize patterns across different industries and roles

### 2.2 Real Dataset (LinkedIn Posting)

#### 2.2.1 Dataset Description

The primary dataset for this project consists of job postings from LinkedIn, collected using the LinkedIn Job Scraper tool (https://github.com/ArshKA/LinkedIn-Job-Scraper). The initial dataset contains 33,246 job postings with 28 columns, including crucial information such as job titles, descriptions, salary ranges, locations, experience levels, and various posting metadata. Each job posting includes structured fields for minimum, maximum, and/or median salary values, though not all postings contain complete salary information.

Key attributes in the dataset include:
- Salary information (minimum, maximum, and median salaries)
- Job metadata (title, description, location, experience level)
- Temporal data (listing time, expiry date)
- Engagement metrics (number of views and applications)
- Job characteristics (work type, remote work status)

#### 2.2.2 Data Preprocessing Steps

The preprocessing pipeline (implemented in `data_preprocess.ipynb`) focused on standardizing salary information and ensuring data quality:

1. **Salary Standardization**
   - Converted all salary values to annual USD
   - Applied conversion multipliers based on pay period:
     * Hourly wages × 2080 (40 hours/week × 52 weeks)
     * Weekly wages × 52
     * Monthly wages × 12
   - Calculated median salary for entries with only min/max values

2. **Data Cleaning**
   - Filtered for USD-denominated salaries only
   - Removed entries without any salary information
   - Eliminated extreme salary outliers (< $10,000 or > $5,000,000 annually)
   - Standardized location information

3. **Missing Value Handling**
   - Retained only entries with at least one valid salary value
   - Preserved rows with partial salary information (min/max only)
   - Maintained records with missing non-critical fields

The final cleaned dataset contains 11,014 job postings, representing approximately 33% of the initial dataset.

#### 2.2.3 Exploratory Analysis Findings

Key insights from the exploratory data analysis include:

1. **Salary Distribution**
   ![Salary Distribution](src/real_data/salary_distribution.png)
   - Median annual salary: $89,634
   - Interquartile range: $59,073 - 130,000
   - Standard deviation: $57,060
   
2. **Experience Level Impact**
   ![Experience Level Impact](src/real_data/salary_by_experience.png)
   - Clear salary progression across experience levels is observed:
     * Entry level: $69,062
     * Mid-Senior level: $115,612
     * Director: $162,099
     * Executive: $173,767
   - Mid-Senior level positions comprise the largest segment (4,150 postings)

3. **Geographic Variations**
   ![Geographic Variations](src/real_data/salary_by_location.png)
   - Highest average salaries:
     * San Francisco Bay Area: $142,967
     * San Francisco: $141,492
     * San Jose: $128,250
   - Most job postings concentrated in major tech hubs

4. **Employment Type Analysis**
   - Full-time positions dominate (9,354 postings)
   - Contract positions show competitive salaries (mean: $101,715)
   - Significant salary gap between full-time and part-time roles

Additional analyses, including detailed visualizations are available in the accompanying Jupyter notebook.

## 3. Model Implementation Description

### 3.1 Approach 1: Fine-tuned BERT Model

#### 3.1.1 Base Model: Pre-trained Transformer (BERT)

#### 3.1.2 Fine-tuning Process

#### 3.1.3 Hyperparameter Tuning

### 3.2 Approach 2: Custom Neural Network with Pre-trained Embeddings

#### 3.2.1 Embedding Layer Setup

#### 3.2.2 Model Architecture

#### 3.2.3 Training Process

## 4. Results and Analysis

### 4.1 Performance on Synthetic Data

### 4.2 Performance on Real Data


## 5. Conclusions

