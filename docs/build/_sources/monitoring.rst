Fairness Monitoring
===================

This python package supports the implementation of a secure two-party protocol for fairness monitoring. The two parties are: 
(1) the service provider operating a job application system, and 
(2) a trusted third party which manages the deposit of protected attributes and runs the computation of fairness metrics.


The setting and protocol
------------------------
This protocol is one of the solutions for fairness monitoring with minimized access to sensitive data. In the setting, we assume two cooperating parties. The first party is the service provider that operates a job application system. The second party is a trusted third party which manages the execution of the protocol. The service provider would like to be able to measure the fairness of their automated system outputs (here, we assume a ranking system ordering applicants/CVs for a given job opening) and hiring decisions for applicants from different protected groups (i.e., groups determined by special categories of data such as ethnicity or sexual orientation). At the same time, the service provider should at no point store or otherwise have access to these special categories of data.

The protocol consists of three operational elements:
(1) A redirection from the service provider's job application system to attribute donation.
(2) An (optional) protected attribute donation by the job applicant and generation of two-party components based on the protected attribute. 
(3) Two-party computation of the fairness metrics.


Protocol Stage 1: Candidate application + redirection to optional attribute deposit
-----------------------------------------------------------------------------------

After the candidate completes a job application, the service provider displays a confirmation screen, which includes a redirection link to a data donation service operated by the third party. The redirection link (or donation service) should allow the third party to (1) unqiuely identify the service provider (e.g., an API key), and (ii) include a service-provider-generated applicant ID, which will allow both parties to uniquely identify the applicant (for a given service provider and job application).

The secure implementation of this data exchange is outside the scope of the library.



Protocol Stage 2: Optional attribute deposit
--------------------------------------------
On the data donation page, an identified applicant (we have a unique service provider ID and applicant ID) can optionally specify the values of selected special categories of data. The third party maintains a catalogue of protected attributes and their possible values. 

Two-party components are then generated independently for each of these optionally specified attributes. For each selected attribute (attribute_name) and applicant-selected value (attribute_value), the third party generates a random secret RS, which is an integer (positive or negative). Two components are then generated: remote_secret_component = (attribute_value - RS) is being sent to the service provider to be stored together with the applicant ID, while local_secret_component = (attribute_value + RS) is being stored locally by the third party together with the applicant ID and the service provider ID. 

The library supports two attribute deposit methods, including (1) back-end distribution: the third party generates and splits encrypted components after receiving user-donated sensitive data. The method is easy to implement, but has stronger trust assumptions in the third party, i.e., the third party is reliable and would not store the raw value before generating the two-party components; and (2) front-end distribution: the two secret components are generated on the front-end and immediately distributed to the service provider and the third party. This method ensures that the third party never accesses the raw sensitive data, but it necessitates more complex coordination.



Protocol Stage 3: Fairness measurement
--------------------------------------
A service provider wishes to measure fairness of their data, system outputs, or hiring decisions. Currently, the library offers an implementation of two-party computations for measuring input fairness, output fairness and outcome fairness. 



Fairness Metrics
----------------
The library offers two-party computations of the following fairness metrics. These metrics allow service providers to measure fairness in data, rankings, or hiring decisions without revealing sensitive attributes, using secure computation via secret sharing with a trusted third party.

Input Fairness (diversity)

- **Pool diversity**: measuring what fraction of the job applicant pool includes members from a given protected group.

	- For instance, if we have a pool of 100 candidates, with 40 women and 60 men, the pool diversity with regard to female candidates is 40/100.

Output Fairness (fair ranking)

- **Group exposure**: measuring what fraction of recruiter attention (or exposure) members of different protected groups receive. In the protocol, the exposure can be specified using theoretical models (e.g., inverse-logarithmic or exponential decay) or manually set based on empirical observations.

	- For example, assume recruiters inspect candidates at the first rank 60% of the time, second rank 30%, third rank 10%, and not at all below that. A recruitment system produces a ranking: man, woman, woman, man, woman. The exposure of men is 0.6 (from the first position). The exposure of women is 0.3 (second) + 0.1 (third) = 0.4.

- **Top-k fairness**: measuring whether members of different protected groups are fairly represented among the top *k* candidates in a ranking. In the protocol, the top-k fairness can be measured either through group proportion comparisons (e.g., Skew\@k) or group exposure comparisons that account for rank position (e.g., Discounted Representation Difference\@k).

	- For example, suppose the full pool has 5 candidates: man, man, woman, woman, woman (2 men, 3 women). Suppose *k* = 3, and the top 3 by the recruitment system are: man, man, woman. The proportion of women in top 3 is 1/3 = 33.3%, but in the full pool is 3/5 = 60%. This underrepresentation yields a negative skew\@k for women.

Outcome Fairness (fair classification)

- **Demographic parity**: measuring what fraction of applicants from different protected groups receive positive outcomes (e.g., interview or hire), regardless of qualifications.

	- For instance, consider a pool of 300 candidates: 100 women and 200 men. If 50 women are hired (50%) and 120 men are hired (60%), then the demographic parity scores are 0.5 for women and 0.6 for men. Therefore, the hiring process does not satisfy demographic parity.

- **Equal opportunity**: measuring what fraction of *qualified* applicants from different protected groups receive positive outcomes. Qualifications can be determined by human evaluation (e.g., HR professionals).

	- For instance, from the above pool, suppose 80 women and 150 men are qualified. If 50 of the 80 qualified women are hired (62.5%) and 100 of the 150 qualified men are hired (66.7%), then the equal opportunity scores are 0.625 for women and 0.667 for men. Despite equal demographic parity score (50% hired in each group), the system does not satisfy equal opportunity.

All the above metrics support **intersectional fairness** analysis, allowing for evaluations across combinations of protected attributes (e.g., gender + age, or ethnicity + disability). This type of analysis helps identify fairness issues that may affect individuals belonging to multiple protected categories and result in intersectional discrimination.

	- For example, in a pool of 200 candidates, there are 40 older workers (aged 50+), consisting of 10 older women and 30 older men. The pool diversity score for older women is 10/200 = 5%, and for older men is 30/200 = 15%. Comparing these values can reveal disparities affecting individuals who belong to multiple protected groups.



Two-Party Fairness Measurement
------------------------------
The library offers both pseudo two-party computation and strict two-party computation to accommodate different real-world needs.

(1) Pseudo two-party computation: The service provider identifies itself using an API key. It then sends the data for which they want to evaluate fairness: an unordered list of candidates for input fairness measurement, a ranked list of candidates for output fairness measurement, or an unordered list of candidates with hiring decisions for outcome fairness measurement. Each list consists of pairs (user_ID, remote_secret_component), where user_ID is the unique user identifier created at the data deposit time, and remote_secret_component is the two-party component of the protected attribute stored by the service provider. Before computing the fairness metric, the third party recreates the protected attributes of applicants by computing: (remote_secret_component + local_secret_component) / 2. Afterwards, fairness metrics are computed as usual. It balances privacy and usability, but has stronger trust assumptions for the third party, i.e., the third party does not store the raw values of protected attributes after they are recreated.

(2) Strict two-party computation: It shares the same computation logic as pseudo two-party computation. However, instead of directly sharing secret components for the reconstruction of protected attributes, the computation of fairness metrics is carried out jointly using two-party computation techniques without recreating protected attributes. It offers a higher level of privacy protection, as no single party ever reconstructs the raw sensitive attribute. However, it requires tight coordination and synchronization between the two parties during the computation process.



Third Party Trust Assumptions
-----------------------------
Note that both remote_secret_component = (attribute_value - RS) and local_secret_component = (attribute_value + RS) are random numbers. Thus, neither the service provider nor the third party store the raw values of the applicant's protected attributes. The parties can however cooperate: the service provider can send the remote_secret_component to the third party who can recreate the protected attributes by summing remote_secret_component and local_secret_component:  attribute_value - RS + attribute_value + RS = 2 * attribute_value. As a result, raw values of protected attributes pass through the computation executed by the third party. 

The third party has to be trusted:

- At the data deposit time: We need to assume that the third party generates the randomized two-party components from the protected attribute, and then discards the raw value of the attribute.

- After the data deposit time: We need to assume that the third party does not secretly communicate with the service provider to recreate the protected attributes based on the two-party components.




Implementation of the third party
---------------------------------
This library provides implementations of the core fairness monitoring protocol function (two-party fairness metrics, computation of protected attribute multiparty components), as well as empty placeholders for other operational functions that need to be implemented by an actual third party service by inheriting the handler classes provided in the library. These operational functions include: local storage and retrieval of protected attribute components for a given service provider and user (e.g., using a database), and sending protected attribute components to the remote service provider. At the moment, we provide examples implementations using CSV files.




