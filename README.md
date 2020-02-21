# Monte-Carlo-Tree-Search
It's a group project that use Monte Carlo Tree Search to solve a dynamic pricing problem for a boutique store. You can also modify the code to achieve other application by using the Monte Carlo Tree Search.

The background of the project is that:

K-Fashion is a boutique store for women’s fashion apparel located in a big shopping mall at the Causeway Bay. The store is targeting young female white-collar who care less about brand but more about fashion and price.
For the next three-month season, K-Fashion has ordered 200 different stock keeping units (SKUs) from a foreign supplier. Due to the long production and order lead time, K-Fashion can place the order only once. Given the large store traffic at Causeway Bay, the store ordered 10 pieces for each SKU. To simplify the analysis, we can assume that the demand for each SKU is independent and statistically identical.
The challenge for K-Fashion is how to maximize the total revenue, given the fixed amount of inventory over the next three months or 12 weeks. Any unsold inventory after the twelfth week will be discarded with zero salvage value. Your job is to focus on the pricing of three SKUs—A, B, and C, the sales of which are independent of other SKUs. The requirements are that:
1. You must set the same price for all the three SKUs as they differ only in color or size. 
2. The price can be adjusted every Monday. 
3. You must pick a price from the set: {999, 899, 799, 699, 599, 499, 399, 299, 199, 99}.

For simplicity, we can assume that each of the three SKUs has its own buyers and cannot substitute each other. Buyers for each SKU arrive randomly. According to past experience, you know that at most one buyer (i.e., 0 or 1) will show up during a day. If a buyer shows up, her
valuation for the SKU will be random and in general the value decreases over time. For example, during the first week, the maximum price acceptable for a buyer may be $1,000; but in the last week the maximum acceptable price may be only $500. If a buyer shows up but finds
that the price is higher than her valuation, then there is a chance that she will come back during the last (twelfth) week of the season with a new, random valuation.
The data that used to train the Monte-Carlo Tree Search is named "Simulated Data"

