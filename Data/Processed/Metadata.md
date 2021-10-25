nba_logreg.csv :

	- Name : Player name
	- GP : Games played 
	- MIN :	AVG Minutes played per game
	- PTS :	AVG Points made per game
	- FGM :	AVG field goals made per game
	- FGA :	AVG field goals attempted per game
	- FG% :	ratio of field goals made to field goals attempted
	- 3PM : AVG three points made per game
	- 3PA : AVG three points attempted per game
	- 3P% :	ratio of three points made to three points attempted
	- FTM :	AVG free throws made per game
	- FTA : AVG free throws attempted per game
	- FT% : ratio of free throws made to free throws attempted
	- OREB : AVG offensive rebounds made per game
	- DREB : AVG defensive rebounds made per game
	- REB : AVG rebounds made per game	
	- AST :	AVG assists made per game
	- STL : AVG steals made per game
	- BLK :	AVG blocks made per game
	- TOV :	AVG turnovers made per game
	- Target : Target value (player career length >=5 == 1) (player career length < 5 == 0)



nba_logreg_selected.csv :

	- Name : Player name
	- GP : Games played 
	- MIN :	AVG Minutes played per game
	- FGA :	AVG field goals attempted per game
	- FG% :	ratio of field goals made to field goals attempted
	- EFG% : Effective Field Goal Percentage to measure the effectiveness of 2-point shots and 3-point shots = 
		 (AVG field goals made per game + 0.5 * AVG three points made per game) * 100 / AVG field goals attempted per game
	- FTM :	AVG free throws made per game
	- FT% : ratio of free throws made to free throws attempted
	- OREB : AVG offensive rebounds made per game
	- DREB : AVG defensive rebounds made per game
	- AST :	AVG assists made per game
	- STL : AVG steals made per game
	- BLK :	AVG blocks made per game
	- TOV :	AVG turnovers made per game
	- Target : Target value (player career length >=5 == 1) (player career length < 5 == 0)
