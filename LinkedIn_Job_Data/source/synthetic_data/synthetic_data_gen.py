import pandas as pd
from typing import List, Dict, Tuple
import random

class SyntheticJobDataGenerator:
    def __init__(self):
        self.seniority_levels = ['Entry Level', 'Mid Level', 'Senior', 'Lead', 'Executive']
        
        self.skills_by_category = {
            'Software Engineering': [
                'Python', 'Java', 'SQL', 'AWS', 'Docker', 'React', 'Node.js'
            ],
            'Data & AI': [
                'Machine Learning', 'Data Analysis'
            ],
            'Product & Management': [
                'Project Management', 'Agile Methodologies', 'Scrum', 'Kanban',
                'Risk Management'
            ],
            'Design': [
                'UX Design', 'UI Design'
            ],
            'Finance': [
                'Financial Modeling', 'Investment Banking', 'Equity Research'
            ],
            'Healthcare': [
                'Healthcare IT', 'Clinical Trials', 'EHR Systems'
            ],
            'Operations': [
                'Supply Chain Management', 'Manufacturing Processes', 'Logistics',
                'Transportation Planning'
            ],
            'Business': [
                'Business Development', 'Retail Sales', 'Customer Service',
                'Marketing Strategy', 'Real Estate Law'
            ]
        }
        self.all_skills = [skill for skills in self.skills_by_category.values() for skill in skills]
        
        self.industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing',
                          'Consulting', 'Hospitality', 'Education', 'Transportation', 'Real Estate']
        self.locations = ['New York', 'San Francisco', 'Seattle', 'Austin', 'Chicago']
        
        # Multipliers and ranges
        self.base_salary_ranges = {
            'Entry Level': (50000, 80000),
            'Mid Level': (80000, 120000),
            'Senior': (120000, 180000),
            'Lead': (150000, 220000),
            'Executive': (200000, 350000)
        }
        
        self.location_multipliers = {
            'New York': 1.2,
            'San Francisco': 1.3,
            'Seattle': 1.15,
            'Austin': 1.0,
            'Chicago': 1.1
        }
        
        self.industry_multipliers = {
            'Technology': 1.2,
            'Finance': 1.15,
            'Healthcare': 1.0,
            'Retail': 0.9,
            'Manufacturing': 0.95,
            'Consulting': 1.1,
            'Hospitality': 0.85,
            'Education': 0.8,
            'Transportation': 0.9,
            'Real Estate': 0.95
        }
        
        self.skill_category_multipliers = {
            'Software Engineering': 1.25,
            'Data & AI': 1.3,
            'Product & Management': 1.15,
            'Design': 1.1,
            'Finance': 1.2,
            'Healthcare': 1.1,
            'Operations': 1.05,
            'Business': 1.0
        }

        # Templates for job description generation
        self.description_templates = [
            "We are seeking a {seniority} {role} to join our {industry} team. " +
            "The ideal candidate will have experience with {skills}. " +
            "This position is based in {location} and offers competitive compensation.",
            
            "Join our growing {industry} company as a {seniority} {role}. " +
            "Required technical skills: {skills}. " +
            "Position located in {location} with excellent benefits.",
            
            "Exciting opportunity for a {seniority} {role} in {location}. " +
            "Our {industry} division is expanding and needs someone skilled in {skills}. " +
            "Competitive salary and comprehensive benefits package."
        ]
        
        self.role_templates = [
            "Software Engineer", "Data Scientist", "Product Manager", "Business Analyst",
            "Project Manager", "Sales Manager", "Marketing Specialist", "UX Designer",
            "Financial Analyst", "Operations Manager", "Supply Chain Analyst",
            "Customer Success Manager", "HR Generalist", "Quality Assurance Engineer"
        ]

    def _get_skill_categories(self, skills: List[str]) -> List[str]:
        """Identify which categories the given skills belong to."""
        categories = []
        for category, category_skills in self.skills_by_category.items():
            if any(skill in category_skills for skill in skills):
                categories.append(category)
        return categories

    def _generate_related_skills(self, num_skills: int) -> List[str]:
        """Generate a set of related skills by first selecting categories."""
        # Select 1-3 skill categories
        num_categories = random.randint(1, min(3, len(self.skills_by_category)))
        selected_categories = random.sample(list(self.skills_by_category.keys()), num_categories)
        
        # Pool available skills from selected categories
        available_skills = []
        for category in selected_categories:
            available_skills.extend(self.skills_by_category[category])
        
        # Select skills (if not enough skills in selected categories, sample from all skills)
        if len(available_skills) < num_skills:
            return random.sample(self.all_skills, num_skills)
        return random.sample(available_skills, num_skills)

    def _generate_salary(self, seniority: str, location: str, 
                        industry: str, skills: List[str]) -> Tuple[float, float]:
        """Generate salary range based on various factors."""
        base_min, base_max = self.base_salary_ranges[seniority]
        
        # Basic multipliers
        location_mult = self.location_multipliers[location]
        industry_mult = self.industry_multipliers[industry]
        skill_mult = 1 + (len(skills) * 0.03)  # Each skill adds 3% to salary
        
        # Calculate skill category multiplier
        categories = self._get_skill_categories(skills)
        category_mult = max([self.skill_category_multipliers[cat] for cat in categories], default=1.0)
        
        # Calculate final salary range
        min_salary = base_min * location_mult * industry_mult * skill_mult * category_mult
        max_salary = base_max * location_mult * industry_mult * skill_mult * category_mult
        
        # Add random variation (±5%)
        min_salary *= random.uniform(0.95, 1.05)
        max_salary *= random.uniform(0.95, 1.05)
        
        return round(min_salary, 2), round(max_salary, 2)

    def _generate_description(self, seniority: str, role: str, 
                            location: str, industry: str, skills: List[str]) -> str:
        """Generate a job description using templates."""
        template = random.choice(self.description_templates)
        
        # Format skills list properly
        if len(skills) > 1:
            skills_str = ", ".join(skills[:-1]) + " and " + skills[-1]
        else:
            skills_str = skills[0]
        
        # Add skill categories to enhance description
        categories = self._get_skill_categories(skills)
        category_str = " and ".join(categories) if categories else "general"
        
        # Enhanced templates with skill categories
        description = template.format(
            seniority=seniority,
            role=role,
            location=location,
            industry=industry,
            skills=skills_str
        )
        
        # Add category-specific additional sentences
        if categories:
            category_sentences = [
                f" Strong background in {category_str} is essential.",
                f" This role focuses on {category_str} skills.",
                f" Expert knowledge of {category_str} is required."
            ]
            description += random.choice(category_sentences)
        
        return description

    def generate_dataset(self, num_samples: int) -> pd.DataFrame:
        """Generate a synthetic dataset with specified number of samples."""
        data = []
        
        for _ in range(num_samples):
            # Generate basic attributes
            seniority = random.choice(self.seniority_levels)
            role = random.choice(self.role_templates)
            location = random.choice(self.locations)
            industry = random.choice(self.industries)
            
            # Generate related skills (3-6 skills)
            skills = self._generate_related_skills(random.randint(3, 6))
            
            # Get categories for these skills
            skill_categories = self._get_skill_categories(skills)
            
            # Generate salary range
            min_salary, max_salary = self._generate_salary(
                seniority, location, industry, skills
            )
            
            # Generate job description
            description = self._generate_description(
                seniority, role, location, industry, skills
            )
            
            # Create record
            record = {
                'title': f"{seniority} {role}",
                'location': location,
                'industry': industry,
                'skills': skills,
                'skill_categories': skill_categories,
                'description': description,
                'min_salary': min_salary,
                'max_salary': max_salary,
                'med_salary': (min_salary + max_salary) / 2
            }
            
            data.append(record)
            
        return pd.DataFrame(data)

    def get_data_generation_rules(self) -> Dict:
        """Return the rules used for data generation."""
        return {
            'base_salary_ranges': self.base_salary_ranges,
            'location_multipliers': self.location_multipliers,
            'industry_multipliers': self.industry_multipliers,
            'skill_category_multipliers': self.skill_category_multipliers,
            'skill_multiplier': '3% per skill',
            'random_variation': '±5% random noise'
        }


generator = SyntheticJobDataGenerator()
data = generator.generate_dataset(1000)
data.to_csv('./synthetic_job_data.csv', index=False)
