# Pool profiling notes (STEP 2)

- Candidates streamed: **100,000**
- Reference now = max(last_active_date): **2026-05-27**
- Mean years_of_experience: **7.17**

## Sentinel / sparsity rates (→ NaN + has_* indicators)
- no GitHub (github_activity_score == -1): 64,637 (64.6%)
- no prior offers (offer_acceptance_rate == -1): 59,554 (59.6%)
- empty skill_assessment_scores: 75,756 (75.8%)
- null education grade (rows): 0
- unknown education tier (rows): 0
- skills that are expert + 0 months (honeypot tell, rows): 84

## Company-type coverage (current employer, by lexicon)
- unknown: 60,982 (61.0%)
- services: 31,197 (31.2%)
- product: 7,821 (7.8%)
- founding_years.csv covers **0** current-company mentions (0.0% of pool)

## Top 60 companies (current + history mentions)
- Infosys (31,312) [services] —
- Wipro (31,248) [services] —
- Wayne Enterprises (31,127) [unknown] —
- Initech (31,118) [unknown] —
- Pied Piper (31,114) [unknown] —
- Acme Corp (31,036) [unknown] —
- Globex Inc (30,963) [unknown] —
- TCS (30,934) [services] —
- Hooli (30,887) [unknown] —
- Dunder Mifflin (30,883) [unknown] —
- Stark Industries (30,847) [unknown] —
- Swiggy (4,307) [product] —
- Razorpay (4,172) [product] —
- CRED (4,165) [product] —
- Capgemini (4,160) [services] —
- Accenture (4,145) [services] —
- HCL (4,144) [services] —
- Zomato (4,109) [product] —
- Mindtree (4,104) [services] —
- Cognizant (4,076) [services] —
- Flipkart (4,053) [product] —
- Tech Mahindra (4,005) [services] —
- Mphasis (3,975) [services] —
- Meesho (570) [product] —
- Nykaa (550) [product] —
- InMobi (548) [unknown] —
- Zoho (517) [product] —
- Ola (514) [product] —
- Vedantu (513) [unknown] —
- BYJU'S (512) [unknown] —
- PolicyBazaar (510) [product] —
- Paytm (504) [product] —
- Freshworks (501) [product] —
- upGrad (493) [unknown] —
- PharmEasy (486) [unknown] —
- PhonePe (480) [product] —
- Dream11 (478) [product] —
- Unacademy (476) [product] —
- Genpact AI (123) [services] —
- Glance (112) [unknown] —
- Rephrase.ai (109) [unknown] —
- Sarvam AI (108) [unknown] —
- Aganitha (106) [unknown] —
- Niramai (105) [unknown] —
- Saarthi.ai (99) [unknown] —
- Krutrim (96) [unknown] —
- Wysa (95) [unknown] —
- Mad Street Den (94) [unknown] —
- Haptik (94) [unknown] —
- Verloop.io (92) [unknown] —
- Observe.AI (85) [unknown] —
- Yellow.ai (84) [unknown] —
- Locobuzz (81) [unknown] —
- Google (20) [product] —
- Netflix (19) [product] —
- Amazon (18) [product] —
- Meta (18) [product] —
- Salesforce (17) [product] —
- Microsoft (15) [product] —
- Uber (12) [product] —

## Top 40 current industries
- IT Services: 29,881
- Software: 22,417
- Manufacturing: 22,305
- Conglomerate: 7,571
- Paper Products: 7,467
- Fintech: 2,808
- Food Delivery: 2,514
- E-commerce: 1,529
- Consulting: 1,274
- EdTech: 610
- SaaS: 328
- AI/ML: 278
- AdTech: 172
- Transportation: 162
- Insurance Tech: 155
- Gaming: 149
- HealthTech: 147
- HealthTech AI: 68
- Conversational AI: 62
- AI Services: 42
- Voice AI: 31
- Internet: 22
- Media: 6
- Consumer Electronics: 2

## Top 60 skills (raw names)
- HTML: 12,246 [—]
- Databricks: 12,244 [—]
- Redux: 12,222 [—]
- Terraform: 12,187 [—]
- Angular: 12,173 [—]
- Figma: 12,157 [—]
- Salesforce CRM: 12,157 [—]
- Vue.js: 12,142 [—]
- Sales: 12,138 [—]
- Accounting: 12,136 [—]
- Agile: 12,135 [—]
- Kafka: 12,114 [—]
- Excel: 12,109 [—]
- BigQuery: 12,108 [—]
- CI/CD: 12,108 [—]
- Project Management: 12,106 [—]
- Airflow: 12,105 [—]
- AWS: 12,104 [—]
- Flask: 12,104 [—]
- Scrum: 12,083 [—]
- Illustrator: 12,072 [—]
- Kubernetes: 12,071 [—]
- ETL: 12,068 [—]
- CSS: 12,065 [—]
- Docker: 12,062 [—]
- Next.js: 12,058 [—]
- Apache Beam: 12,054 [—]
- Java: 12,049 [—]
- Go: 12,049 [—]
- TypeScript: 12,048 [—]
- JavaScript: 12,047 [—]
- dbt: 12,046 [—]
- REST APIs: 12,040 [—]
- Spark: 12,038 [—]
- Marketing: 12,037 [—]
- Tally: 12,030 [—]
- GraphQL: 12,027 [—]
- Snowflake: 12,027 [—]
- Webpack: 12,026 [—]
- Six Sigma: 11,991 [—]
- SEO: 11,990 [—]
- SAP: 11,989 [—]
- GCP: 11,983 [—]
- PostgreSQL: 11,983 [—]
- Rust: 11,960 [—]
- Apache Flink: 11,958 [—]
- gRPC: 11,957 [—]
- Content Writing: 11,948 [—]
- SQL: 11,935 [—]
- Hadoop: 11,931 [—]
- Redis: 11,928 [—]
- Tailwind: 11,917 [—]
- Photoshop: 11,917 [—]
- FastAPI: 11,917 [—]
- Microservices: 11,909 [—]
- PowerPoint: 11,908 [—]
- Spring Boot: 11,906 [—]
- Data Pipelines: 11,905 [—]
- Django: 11,899 [—]
- MongoDB: 11,841 [—]

## Locations / countries / sizes
Top 25 cities: bhubaneswar(4321), noida(4283), hyderabad(4283), jaipur(4268), bangalore(4238), kolkata(4230), indore(4198), pune(4186), chennai(4164), delhi(4161), trivandrum(4151), ahmedabad(4143), chandigarh(4128), coimbatore(4113), vizag(4093), kochi(4073), mumbai(4043), gurgaon(4037), sydney(2579), san francisco(2536), austin(2531), new york(2518), toronto(2506), london(2472), berlin(2469)
Countries: India(75113), USA(9978), Australia(2579), Canada(2506), UK(2472), Germany(2469), Singapore(2453), UAE(2430)
Company sizes: 10001+(40464), 1001-5000(18201), 201-500(15096), 51-200(7727), 11-50(7568), 501-1000(7525), 5001-10000(3419)
Proficiency: intermediate(470309), beginner(379097), advanced(109585), expert(1311)
