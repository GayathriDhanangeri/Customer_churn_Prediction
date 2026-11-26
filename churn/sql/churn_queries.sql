-- Project 02 - Churn analytics SQL (tested on SQLite; SQL Server and PostgreSQL need minor date-syntax tweaks)
-- Load powerbi/fact_customers.csv into a table named fact_customers first.

-- 1. Reconciliation: row count, churners, churn rate (must match the cleaned Python dataset: 10,000 customers, 3,096 churners)
SELECT COUNT(*) AS customers, SUM(churned) AS churners, ROUND(AVG(churned*1.0),4) AS churn_rate
FROM fact_customers;

-- 2. Churn rate by contract type
SELECT contract_type, COUNT(*) AS customers, ROUND(AVG(churned*1.0),3) AS churn_rate
FROM fact_customers GROUP BY contract_type ORDER BY churn_rate DESC;

-- 3. Churn by tenure band and plan
SELECT tenure_band, plan, COUNT(*) AS customers, ROUND(AVG(churned*1.0),3) AS churn_rate
FROM fact_customers GROUP BY tenure_band, plan ORDER BY tenure_band, plan;

-- 4. Risk-band performance: do higher bands actually churn more?
SELECT risk_band, COUNT(*) AS customers, SUM(churned) AS churners,
       ROUND(AVG(churned*1.0),3) AS actual_churn_rate, ROUND(AVG(risk_score),3) AS avg_predicted_risk
FROM fact_customers
GROUP BY risk_band
ORDER BY CASE risk_band WHEN 'Low' THEN 1 WHEN 'Moderate' THEN 2 WHEN 'High' THEN 3 ELSE 4 END;

-- 5. Share of all churners captured by each band (cumulative, highest risk first)
WITH b AS (
  SELECT risk_band, SUM(churned) AS churners, COUNT(*) AS customers,
         CASE risk_band WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Moderate' THEN 3 ELSE 4 END AS rk
  FROM fact_customers GROUP BY risk_band)
SELECT risk_band, customers, churners,
       ROUND(1.0*SUM(churners) OVER (ORDER BY rk) / SUM(churners) OVER (), 3) AS cumulative_churners_captured,
       ROUND(1.0*SUM(customers) OVER (ORDER BY rk) / SUM(customers) OVER (), 3) AS cumulative_customers_contacted
FROM b ORDER BY rk;

-- 6. Segment summary with monthly revenue at risk (predicted churners only)
SELECT segment, COUNT(*) AS customers, ROUND(AVG(churned*1.0),3) AS actual_churn,
       ROUND(SUM(CASE WHEN predicted_churn=1 THEN monthly_charges ELSE 0 END),0) AS monthly_revenue_at_risk
FROM fact_customers GROUP BY segment ORDER BY monthly_revenue_at_risk DESC;

-- 7. Retention work-queue: top 25 P1/P2 customers by risk x value
SELECT customer_id, segment, retention_priority, ROUND(risk_score,3) AS risk_score, monthly_charges, contract_type, support_contacts_90d
FROM fact_customers
WHERE retention_priority LIKE 'P1%' OR retention_priority LIKE 'P2%'
ORDER BY risk_score * monthly_charges DESC LIMIT 25;

-- 8. Observed cancellations by month (post-snapshot window) and reason
SELECT churn_month, cancel_reason, COUNT(*) AS cancellations
FROM fact_customers WHERE churned = 1
GROUP BY churn_month, cancel_reason ORDER BY churn_month, cancellations DESC;

-- 9. Churn by sign-up cohort (monthly cohorts, last 12 months)
SELECT signup_month, COUNT(*) AS customers, ROUND(AVG(churned*1.0),3) AS churn_rate
FROM fact_customers WHERE signup_month >= '2025-01-01' GROUP BY signup_month ORDER BY signup_month;

-- 10. Support-contact pressure vs churn, by payment method
SELECT payment_method, support_band, COUNT(*) AS customers, ROUND(AVG(churned*1.0),3) AS churn_rate
FROM fact_customers GROUP BY payment_method, support_band ORDER BY payment_method, support_band;
