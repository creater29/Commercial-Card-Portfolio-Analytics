-- Portfolio Performance Analysis for Commercial Cards
-- Key metrics and insights for portfolio management

-- 1. Monthly Transaction Volume and Spend Analysis
WITH monthly_metrics AS (
    SELECT 
        DATE_TRUNC('month', transaction_date) as month,
        COUNT(*) as transaction_count,
        SUM(transaction_amount) as total_spend,
        AVG(transaction_amount) as avg_transaction_amount,
        COUNT(DISTINCT customer_id) as active_customers
    FROM card_transactions 
    WHERE transaction_status = 'APPROVED'
        AND transaction_date >= CURRENT_DATE - INTERVAL '24 months'
    GROUP BY DATE_TRUNC('month', transaction_date)
),
growth_metrics AS (
    SELECT 
        month,
        transaction_count,
        total_spend,
        avg_transaction_amount,
        active_customers,
        LAG(total_spend) OVER (ORDER BY month) as prev_month_spend,
        (total_spend - LAG(total_spend) OVER (ORDER BY month)) / 
            LAG(total_spend) OVER (ORDER BY month) * 100 as spend_growth_pct
    FROM monthly_metrics
)
SELECT 
    month,
    transaction_count,
    total_spend,
    avg_transaction_amount,
    active_customers,
    spend_growth_pct
FROM growth_metrics
ORDER BY month DESC;

-- 2. Customer Segmentation Analysis
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.customer_since_date,
        c.annual_income,
        c.credit_limit,
        SUM(t.transaction_amount) as total_spend,
        COUNT(t.transaction_id) as transaction_count,
        MAX(t.transaction_date) as last_transaction_date,
        CURRENT_DATE - MAX(t.transaction_date) as recency_days
    FROM customer_master c
    LEFT JOIN card_transactions t ON c.customer_id = t.customer_id
    WHERE c.status = 'ACTIVE'
        AND t.transaction_status = 'APPROVED'
        AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY c.customer_id, c.customer_segment, c.customer_since_date, 
             c.annual_income, c.credit_limit
),
segment_performance AS (
    SELECT 
        customer_segment,
        COUNT(*) as customer_count,
        AVG(total_spend) as avg_customer_spend,
        SUM(total_spend) as segment_total_spend,
        AVG(transaction_count) as avg_transactions_per_customer,
        AVG(recency_days) as avg_recency_days,
        COUNT(CASE WHEN recency_days <= 30 THEN 1 END) * 100.0 / COUNT(*) as active_rate_30d
    FROM customer_metrics
    GROUP BY customer_segment
)
SELECT 
    customer_segment,
    customer_count,
    avg_customer_spend,
    segment_total_spend,
    avg_transactions_per_customer,
    avg_recency_days,
    active_rate_30d
FROM segment_performance
ORDER BY segment_total_spend DESC;

-- 3. Geographic Performance Analysis
SELECT 
    c.geographic_region,
    COUNT(DISTINCT c.customer_id) as customer_count,
    SUM(t.transaction_amount) as total_spend,
    AVG(t.transaction_amount) as avg_transaction_amount,
    COUNT(t.transaction_id) as total_transactions,
    SUM(t.transaction_amount) / COUNT(DISTINCT c.customer_id) as spend_per_customer
FROM customer_master c
JOIN card_transactions t ON c.customer_id = t.customer_id
WHERE c.status = 'ACTIVE'
    AND t.transaction_status = 'APPROVED'
    AND t.transaction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY c.geographic_region
ORDER BY total_spend DESC;

-- 4. Merchant Category Analysis
WITH category_analysis AS (
    SELECT 
        merchant_category,
        COUNT(*) as transaction_count,
        SUM(transaction_amount) as total_spend,
        AVG(transaction_amount) as avg_transaction_amount,
        COUNT(DISTINCT customer_id) as unique_customers,
        COUNT(DISTINCT merchant_name) as unique_merchants
    FROM card_transactions
    WHERE transaction_status = 'APPROVED'
        AND transaction_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY merchant_category
)
SELECT 
    merchant_category,
    transaction_count,
    total_spend,
    avg_transaction_amount,
    unique_customers,
    unique_merchants,
    total_spend * 100.0 / SUM(total_spend) OVER () as spend_percentage
FROM category_analysis
ORDER BY total_spend DESC;

-- 5. Customer Lifetime Value Analysis
WITH customer_clv AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.customer_since_date,
        EXTRACT(DAYS FROM (CURRENT_DATE - c.customer_since_date)) / 365.25 as tenure_years,
        SUM(t.transaction_amount) as total_spend,
        COUNT(t.transaction_id) as total_transactions,
        SUM(t.transaction_amount) / NULLIF(EXTRACT(DAYS FROM (CURRENT_DATE - c.customer_since_date)) / 365.25, 0) as annual_spend_rate
    FROM customer_master c
    LEFT JOIN card_transactions t ON c.customer_id = t.customer_id
    WHERE c.status = 'ACTIVE'
        AND t.transaction_status = 'APPROVED'
    GROUP BY c.customer_id, c.customer_segment, c.customer_since_date
),
clv_segments AS (
    SELECT 
        customer_id,
        customer_segment,
        tenure_years,
        total_spend,
        annual_spend_rate,
        CASE 
            WHEN annual_spend_rate >= 50000 THEN 'High Value'
            WHEN annual_spend_rate >= 20000 THEN 'Medium Value'
            WHEN annual_spend_rate >= 5000 THEN 'Low Value'
            ELSE 'Minimal Value'
        END as clv_segment
    FROM customer_clv
    WHERE tenure_years > 0
)
SELECT 
    clv_segment,
    COUNT(*) as customer_count,
    AVG(total_spend) as avg_total_spend,
    AVG(annual_spend_rate) as avg_annual_spend,
    AVG(tenure_years) as avg_tenure_years
FROM clv_segments
GROUP BY clv_segment
ORDER BY avg_annual_spend DESC;

-- 6. Churn Risk Analysis
WITH customer_activity AS (
    SELECT 
        customer_id,
        MAX(transaction_date) as last_transaction_date,
        CURRENT_DATE - MAX(transaction_date) as days_since_last_transaction,
        COUNT(*) as transaction_count_12m,
        SUM(transaction_amount) as total_spend_12m,
        AVG(transaction_amount) as avg_transaction_amount
    FROM card_transactions
    WHERE transaction_status = 'APPROVED'
        AND transaction_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY customer_id
),
churn_risk AS (
    SELECT 
        ca.customer_id,
        c.customer_segment,
        ca.days_since_last_transaction,
        ca.transaction_count_12m,
        ca.total_spend_12m,
        CASE 
            WHEN ca.days_since_last_transaction > 90 THEN 'High Risk'
            WHEN ca.days_since_last_transaction > 60 THEN 'Medium Risk'
            WHEN ca.days_since_last_transaction > 30 THEN 'Low Risk'
            ELSE 'Active'
        END as churn_risk_category
    FROM customer_activity ca
    JOIN customer_master c ON ca.customer_id = c.customer_id
    WHERE c.status = 'ACTIVE'
)
SELECT 
    churn_risk_category,
    COUNT(*) as customer_count,
    AVG(total_spend_12m) as avg_spend_12m,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage_of_portfolio
FROM churn_risk
GROUP BY churn_risk_category
ORDER BY 
    CASE churn_risk_category
        WHEN 'High Risk' THEN 1
        WHEN 'Medium Risk' THEN 2
        WHEN 'Low Risk' THEN 3
        WHEN 'Active' THEN 4
    END;
