# Case Study: Predictive Marketing Model for Retail Food Sector

**This project presents a case study on predictive modeling for marketing campaigns in the retail food sector. In this scenario, the goal is to build a predictive model that helps a well-established food retail company optimize its direct marketing strategies, increasing profitability and better targeting customers likely to purchase specific products. The company is experiencing stagnant profit growth and hopes to improve the performance of its marketing efforts through data-driven decisions.**

In this repository, you will find the detailed steps involved in building a predictive model using customer data, socio-demographic, and firmographic variables. The focus is to identify high-potential customers for the upcoming marketing campaign, minimizing costs and maximizing return on investment.

Continue reading to explore the project breakdown, starting with the business context, marketing department's objective, data description, and model approach.

- *The Company* -
Consider a well-established company operating in the retail food sector. Presently they have around 
several hundred thousand registered customers and serve almost one million consumers a year. 
They sell products from 5 major categories: wines, rare meat products, exotic fruits, specially 
prepared fish and sweet products. These can further be divided into gold and regular products. The 
customers can order and acquire products through 3 sales channels: physical stores, catalogs and 
company’s website. Globally, the company had solid revenues and a healthy bottom line in the past 
3 years, but the profit growth perspectives for the next 3 years are not promising... For this reason, 
several strategic initiatives are being considered to invert this situation. One is to improve the 
performance of marketing activities, with a special focus on marketing campaigns. 

- *The marketing department* - 
The marketing department was pressured to spend its annual budget more wisely. The CMO 
perceives the importance of having a more quantitative approach when taking decisions, reason why 
a small team of data scientists was hired with a clear objective in mind: to build a predictive model 
which will support direct marketing initiatives. Desirably, the success of these activities will prove the 
value of the approach and convince the more skeptical within the company. 

- *The objective* - 
The objective of the team is to build a predictive model that will produce the highest profit for the 
next direct marketing campaign, scheduled for the next month. The new campaign, sixth, aims at 
selling a new gadget to the Customer Database. To build the model, a pilot campaign involving 2.240 
customers was carried out. The customers were selected at random and contacted by phone 
regarding the acquisition of the gadget. During the following months, customers who bought the 
offer were properly labeled. The total cost of the sample campaign was 6.720MU and the revenue 
generated by the customers who accepted the offer was 3.674MU. Globally the campaign had a 
profit of -3.046MU. The success rate of the campaign was 15%. The objective is of the team is to 
develop a model that predicts customer behavior and to apply it to the rest of the customer base. 
Hopefully, the model will allow the company to cherry pick the customers that are most likely to 
purchase the offer while leaving out the non-respondents, making the next campaign highly 
profitable. Moreover, other than maximizing the profit of the campaign, the CMO is interested in 
understanding to study the characteristic features of those customers who are willing to buy the 
gadget. 

- *The data*  -
The data set contains socio-demographic and firmographic features about 2.240 customers who 
were contacted. Additionally, it contains a flag for those customers who responded the campaign, 
by buying the product.

Variables in data frame:

- AcceptedCmp1 : 1 if customer accepted order in 1st campaign, 0 otherwise
- AcceptedCmp2 : 1 if customer accepted order in 2nd campaign, 0 otherwise
- AcceptedCmp3 : 1 if customer accepted order in 3rd campaign, 0 otherwise
- AcceptedCmp4 : 1 if customer accepted order in 4th campaign, 0 otherwise
- AcceptedCmp5 : 1 if customer accepted order in 5th campaign, 0 otherwise
- Complain : 1 if customer complained in the last 2 years
- Dt_Customer : date of custoemr enrollement with the company
- Education : customer level of education
- ID : primary key
- Income : customer yearly household income
- Kidhome : number of small children in customer's household
- Marital_Status : customer marital status
- MntFishProducts : amount spent in fish products in the last 2 years
- MntFruits : amount spent in fruits products in the last 2 years
- MntGoldProds : amount spent in gold (premium - may be either meat, fruit, fish, sweet or wine) products in the last 2 years
- MntMeatProducts : amount spent in meat products in the last 2 years
- MntSweetProducts : amount spent in sweet products in the last 2 years
- MntWines : amount spent in wine products in the last 2 years
- NumCatalogPurchases : number of purchases made using catalogue
- NumDealsPurchases : 
- NumStorePurchases : number of purchases made directly in a physical store
- NumWebPurchases : number of purchases made through company website
- NumWebVisitsMonth : number of visits to company's web site in the last month
- Recency : number of days since the last purchase
- Response (target): 1 if customer accepted the offer in the last campaign, 0 otherwise
- Teenhome : number of teens in customer's household
Year_Birth : Year the client has been born
