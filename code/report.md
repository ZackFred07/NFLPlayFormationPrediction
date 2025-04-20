# NFL Snap prediction

## Dataset

This dataset is broken down by `games.csv`, `player_play.csv`, `players.csv`, `plays.csv` and `tracking_week_*.csv`

### Parameter Selection

These are the selected features and labels with identifiers, all numerical values and not-useful categorical values are eliminated.

Values that are global time series wise are in `games.csv` and `plays.csv` 

Values that are local are in `player_play.csv`, `players.csv`, and `tracking_week_*.csv`

`games.csv`
- gameId: Identifer
- season: Identifer
- week: Identifer
- gameDate: Ordinal feature
- gameTimeEastern: Ordinal feature
- ~~homeTeamAbbr~~ We can use the tracking file
- ~~visitorTeamAbbr~~ We can use the tracking
- ~~homeFinalScore~~
- ~~visitorFinalScore~~

`player_play.csv`
- gameId: Identifer
- playId: Identifer
- nflId: Identifer
- teamAbbr: Identifer
- ~~hadRushAttempt~~
- ~~rushingYards~~
- ~~hadDropback~~
- ~~passingYards~~
- ~~sackYardsAsOffense~~
- ~~hadPassReception~~
- ~~receivingYards~~
- ~~wasTargettedReciever~~
- ~~YardageGainedAfterTheCatch~~
- ~~fumbles~~
- ~~fumbleLost~~
- ~~fumbleOutOfBounds~~
- ~~assistedTackle~~
- ~~forceFumbleAsDefense~~
- ~~halfSackYardsAsDefense~~
- ~~passDefensed~~
- ~~quarterbackHit~~
- ~~sackYardsAsDefense~~
- ~~safetyAsDefense~~
- ~~soloTackle~~
- ~~tackleAssist~~
- ~~tackleForALoss~~
- ~~tackleForALossYardage~~
- ~~hadInterception~~
- ~~interceptionYards~~
- ~~fumbleRecoveries~~
- ~~fumbleRecoveryYards~~
- ~~penaltyYards~~
- ~~penaltyNames~~
- ~~wasInitialPassRusher~~
- ~~causedPressure~~
- ~~timeToPressureAsPassRusher~~
- ~~getOffTimeAsPassRusher~~
- ~~inMotionAtBallSnap~~
- ~~shiftSinceLineset~~
- ~~motionSinceLineset~~
- wasRunningRoute: Will aid in label 
- routeRan: Will need to be modified to where its left to right by position as category or concatenation, 
- ~~blockedPlayerNFLId1~~
- ~~blockedPlayerNFLId2~~
- ~~blockedPlayerNFLId3~~
- ~~pressureAllowedAsBlocker~~
- ~~timeToPressureAllowedAsBlocker~~
- ~~pff_defensiveCoverageAssignment~~
- ~~pff_primaryDefensiveCoverageMatchupNflId~~
- ~~pff_secondaryDefensiveCoverageMatchupNflId~~

`players.csv`
- nflId: Identifer
- height: String value that needs to be a ratio integer
- weight: Ratio value used as feature
- birthDate: Used to calculate age
- ~~collegeName~~
- position: Nominal value used as feature
- ~~displayName~~


`plays.csv`
- gameId: Identifer
- playId: Identifer
- ~~playDescription~~
- quarter: feature
- down: feature
- yardsToGo: feature
- possessionTeam: feature
- defensiveTeam: feature
- yardlineSide: feature
- ~~yardlineNumber: feature~~ absolute yardline number is used
- gameClock: feature
- preSnapHomeScore: feature
- preSnapVisitorScore: feature
- ~~playNullifiedByPenalty~~
- absoluteYardlineNumber: feature
- ~~preSnapHomeTeamWinProbability~~
- ~~preSnapVisitorTeamWinProbability~~
- ~~expectedPoints~~
- offenseFormation: label
- receiverAlignment: label
- ~~playClockAtSnap~~
- ~~passResult~~
- ~~passLength~~
- ~~targetX~~
- ~~targetY~~
- playAction: label
- dropbackType: label
- ~~dropbackDistance~~
- passLocationType: label
- ~~timeToThrow~~
- ~~timeInTackleBox~~
- ~~timeToSack~~
- passTippedAtLine: label
- unblockedPressure: label
- qbSpike: label
- qbKneel: label
- qbSneak: label
- rushLocationType: label
- ~~penaltyYards~~
- ~~prePenaltyYardsGained~~
- ~~yardsGained~~
- ~~homeTeamWinProbabilityAdded~~
- ~~visitorTeamWinProbilityAdded~~
- ~~expectedPointsAdded~~
- isDropback: label
- pff_runConceptPrimary: label
- pff_runConceptSecondary: label
- pff_runPassOption: label
- pff_passCoverage: label
- pff_manZone: label


`tracking_week_*.csv`
- gameId: Identifer
- playId: Identifer
- nflId: Identifer
- ~~displayName~~
- frameId: Identifer
- frameType: Identifer
- time: Identifer, mostly useful for visualization
- ~~jerseyNumber~~
- club: feature
- playDirection: used for feature engineering
- x: feature
- y: feature
- s: feature
- a: feature
- dis: feature
- o: feature
- dir: feature
- event: Identifer, useful for the evaluation process

### Features

global_features
- ~~gameId~~
- ~~playId~~
- quarter: Ordinal[1,2,3,4]
- down: Ordinal[1,2,3,4]
- yardsToGo: Ratio Integer
- PossessionTeam: Nominal
- defensiveTeam: Nominal
- gameClock: TODO: need to change to seconds
- preSnapHomeScore: Ratio Integer
- preSnapVisitorScore: Ratio Integer
- absoluteYardlineNumber: Ratio Integer
- week: Ordinal[1,2,3,4,5,6,7,8,9]
- ~~gameDate~~
- gameTimeEastern: Ratio seconds TODO: need to change to seconds
- ~~playDirection~~

local_features
- ~~gameId~~
- ~~playId~~
- ~~nflId~~
- ~~frameId~~,
- ~~frameType~~
- ~~time~~: 
- ~~club~~ Should be assumed from global value and position
- ~~playDirection~~
- x: Ratio
- y: Ratio
- s: Ratio
- a: Ratio
- dis: Ratio
- ~~o~~
- ~~dir~~
- ~~event~~
- height: Ratio
- weight: Ratio
- ~~birthDate~~
- position: Nominal
- sin_o: Ratio
- cos_o: Ratio
- sin_dir: Ratio
- cos_dir: Ratio
- rel_x
- rel_y

### Labels

labels
- routeRanQB0: Nominal
- routeRanWR0: Nominal
- routeRanWR1: Nominal
- routeRanWR2: Nominal
- routeRanWR3: Nominal
- routeRanWR4: Nominal
- routeRanRB0: Nominal
- routeRanRB1: Nominal
- routeRanRB2: Nominal
- routeRanTE0: Nominal
- routeRanTE1: Nominal
- routeRanTE2: Nominal
- offenseFormation: Nominal
- receiverAlignment: Nominal
- playAction: Boolean
- dropbackType: Nominal
- passLocationType: Nominal
- passTippedAtLine: Boolean
- unblockedPressure: Boolean
- qbSpike: Boolean
- qbKneel: Boolean
- qbSneak: Boolean
- rushLocationType: Nominal
- isDropback: Boolean
- pff_runConceptPrimary: Nominal
- pff_runConceptSecondary: Nominal
- pff_runPassOption: Boolean
- pff_passCoverage: Nominal
- pff_manZone: Nominal

### Statistics

## Data Preprocessing

### Transformations

### 2D Tensors

### Tabular

### Graph

## Model Selection

### ConvLSTM

### C3D

### HeatmapTransformer

### ViT

### VidViT

## Train

### Normal Training

### Truncated Training

## Evaluation

### Basic Metrics

- Accuracy
- Precision
- Recall
- F1

### Truncated Metrics

-  
- 
- 
- 