---
title: Can AI agent find the best place to visit in the commercial district? - Deep Reinforcement learning approach
date: 2020-10-20 17:00:00 -0400
classes: wide
layout: single
categories: Itaewon
---
 
# Introduction to Itaewon Instagram contents map

Before visiting certain areas, we routinely visit restaurants and cafes after searching for reviews of social media, and go to the store that is expected to give you the highest satisfaction through a search.
But is it really the best choice to visit the shop using the results of search? 
If the store I found was famous for a short time or if I didn't search enough, I wouldn't be able to find the most famous store in the area.
Rather than spending more time on the Internet to find the best place to visit, I think we can solve this problem if we know the distribution of Instagram contents throughout commercial districts.

This map expresses the Instagram popularity of Itaewon commercial district which is located in Itaewon([Map 1](https://geoai-rl.github.io/ITW_baseMap_20201011.html), [Map 2](https://geoai-rl.github.io/ITW_timeInstagramCummulative.html)), Yongsan-gu, Seoul, South Korea. It is one of the leading business districts in Seoul. 
Also not only represent Instagram distribution, but train the agent which visited the area based on the Instagram popularity using deep reinforcement learning([Map 3](https://geoai-rl.github.io/SimulationResult_20201012.html)).
At first, an agent randomly interacted with the map, however, after simulation, the agent could finally learn the optimized policy to find the most famous areas based on given data.
For making this map, 20 computers were parallelized to collect Instagram data. Names of stores in Itaewon area were used as queries for search on Instagram and parsed Instagram information. Through this process, Instagram contents that has been searched between January 1, 2013, and August 19, 2019 (approximately 2167 days) were saved in the database.

[[Map 1]](https://geoai-rl.github.io/ITW_baseMap_20201011.html) shows Instagram contents that were generated on August 19, 2019, the last day contents were posted.
It is hard to find the trend of overall online popularity in the Itaewon commercial districts if only data at a certain point in time is used.
To find timeseries trend in the Itaewon area, spatiotemporal distribution of the Instagram contents in the Itaewon commercial district was expressed in [[Map 2]](https://geoai-rl.github.io/ITW_timeInstagramCummulative.html).
In 2013, when Instagram was launched in Korea, little contents were created in Itaewon area, but as it goes to August 2019, it can be seen that amount of contents regarding the research area have been gradually increased.

In particular, the distribution of contents can identify the active areas in the region at a certain time.
To describe the popularity distribution in the Itaewon area, I trained artificial intelligence that can support spatial decisions by utilizing data. Instagram contents are spatiotemporal data that have a significant impact on the floating population([Jang, et.al, 2020](https://github.com/geoai-rl/geoai-rl.github.io/blob/main/spatialpanel_instagram_1023.pdf))

The Instagrm data which are heterogeneously distributed in the area was converted into the learning environment, and the agent which interacted with the environment were trained using deep reinforcement learning.
As a result of the study, we were able to train the agent who learned the optimal policy of visiting the most famous area.[[Map 3]](https://geoai-rl.github.io/SimulationResult_20201012.html)

# [Map 1](https://geoai-rl.github.io/ITW_baseMap_20201011.html) : Itaewon Instagram contents map(Static Map)
* *If you want to see the map bigger, click 'Map 1'*

<center><img src="/assets/basemapimage.jpg" width="600" height="500"></center>
<center>Itaewon Instagram contents map 1 example</center>
* *Colors range from blue to red*
* *Blue color represents the lowest frequency, on the other hand, red color represents the highest frequency*
<br/>

[[Map 1]](https://geoai-rl.github.io/ITW_baseMap_20201011.html) shows Instagram data, store locations, research areas, and simulation scope data on the map.
* HeatMap (Instagram Buzz) layer: Instagram data is displayed in HeatMap (Instagram Buzz) layer. It was expressed in HeatMap to see the distribution of Instagram contents within the Itaewon district. Blue color represents the lowest frequency, on the other hand, red color represents the highest frequency
* BUILDING (341) layer: The location of stores in Itaewon area can be found in BUILDING (341) layer. A total of 341 shops are displayed in research areas. Markers are marked on the coordinates where the store is located, and you can check the total number of "accumulated" Instagram contents by clicking on them.
* Simulation Environment (Grid) Layer: The simulation Environment (Grid) Layer indicates the environment in which deep reinforcement learning is applied. Around Itaewon Station, the range of 1000㎡ was used as the simulation environment, and the environment of 1,000㎡ consisted of 10,000 grids(10㎡).
* Research Area layer: Research Area layer indicates the research areas which consist of three administrative areas(Yongsan-dong 2-ga, Itaewon-dong, Hannam-dong) where Itaewon commercial districts are formed.

# [Map 2](https://geoai-rl.github.io/ITW_timeInstagramCummulative.html) : Itaewon Instagram contents map(Dynamic Map)
* *If you want to see the map bigger, click 'Map 2'*

<center><img src="/assets/map2Example.gif" width="600" height="500"></center>
<center>Map 2 example</center>
* *Colors range from blue to red*
* *Blue color represents the lowest frequency, on the other hand red color represents the highest frequency*
<br/>

[[Map 2]](https://geoai-rl.github.io/ITW_timeInstagramCummulative.html) shows a chronological display of Instagram contents that were posted between January 1, 2013, and August 19, 2019.
The cumulative sum of contents in the area was expressed in a timely order to identify the activated areas of the Itaewon commercial district.
The distribution of contents allows us to identify the active areas at a specific point in time.

# [Map 3](https://geoai-rl.github.io/SimulationResult_20201012.html) : Deep Reinforcement learning simulation result(dynamic map)
* *If you want to see the map bigger, click 'Map 3'*
* *Click 'Map 3', then press the 3rd button on the bottom left for playback*

<center><img src="/assets/images//reinforcementlearning.jpg" width="500" height="500"></center>
<center>Reinforcement learning process</center>
<br/>

The process in which the reinforcement learning agent proceeds learning is as shown in the image above. 
The agent selects an action according to the state of the environment and receives a reward from the environment based on the action.

<center><img src="/assets/images/grid.jpg" width="300" height="300"/> <img src="/assets/gridSimulation.gif" width="300" height="300"></center>
<center>Left : Map 1 - Simulation Environment(Grid)</center>
<center>Right : Deep Reinforcement Learning Simulation in Grid World</center>
<center>(Right image corresponds to the same coordinates on the left image)</center>
* *circle: Agent who is interacting with the environment*
* *Red grid: Red grids represent the place where shops are located*
<br/>

[[Map 3]](https://geoai-rl.github.io/SimulationResult_20201012.html) demonstrates the training process using deep reinforcement learning.
In deep reinforcement learning simulation, the agent receives high rewards for visiting areas with a lot of Instagram posts, otherwise, the agent will not be rewarded. Through learning, the agent finds the optimal policy to visit the most famous area in Itaewon.
As can be seen in [Map 1](https://geoai-rl.github.io/ITW_baseMap_20201011.html)'s 'Simulation Environment (Grid) Layer', a grid environment was formed to respond to the actual space range, and Instagram contents that were generated in the areas corresponding to the space range of each grid were aggregated into each grid.

<center><img src="/assets/simulationResult.gif" width="600" height="500"></center>
<center>Map 3 - Deep Reinforcement Learning Simulation in Real World</center>
* *Colors range from blue to red* 
* *Blue color represents the lowest frequency, on the other hand red color represents the highest frequency*
* *The most visited areas become red and relatively the least visited areas become blue*
<br/>

The results of simulation between August 2018 and August 2019 are as shown in [Map 3](https://geoai-rl.github.io/SimulationResult_20201012.html).
It is possible to check the frequency of the agent's visits according to simulation.
Areas which were highly visited by the agent were shown red in Heatmap, otherwise blue.
The agent randomly visited areas in the early training stages, but as training progressed, it can learn the optimal policy to visit the most famous areas, and consequently the agent is converged into the areas where the highest rewards can be obtained.

Based on the results of Itaewon Instagram Contents Map, visitors to the commercial district can plan an optimal plan to visit the best location where famous shops are clustered.
It is also expected to provide useful information to small business owners who run business in the research area.
We can identify active commercial districts at certain time, and we can also use the trained agent to predict areas which will be popular in the future.
It is expected that this research can be used to support spatial decisions for small business owners when they select new store locations.
Furthermore, we look forward to the possibility of developing the vanishing process of stores due to higher rents, which can predict and respond to gentrification in future researches.

* This post is based on the papers below
<br/>
*(in progress) Jang, J., & Choi, J. (2020). Predicting pedestrian behaviors in Itaewon commercial district using user generated contents : deep reinforcement learning approach.*
<br/>
*Jang, J. (1st author), Kim, M., Choi, J., (2020), A Study on the Relationship between Foot Traffic and Instagram Contents Using Spatial Panel Model, Korea Society for Geospatial Information Science ([in Korean with English abstract](https://github.com/geoai-rl/geoai-rl.github.io/blob/main/spatialpanel_instagram_1023.pdf)))*
