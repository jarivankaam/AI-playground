using System;
using System.Collections.Generic;
using TorchSharp;


int[,] maze1 = {
        { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 },
        { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 },
        { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 },
        { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 },
        { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 },
        { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 },
        { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 },
        { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 },
        { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 },
        { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 },
        { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 },
        { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }
    };

string[] actions = { "up", "down", "left", "right" };
int[,] rewards;
torch.Tensor qValues;

const int WALL_REWARD_VALUE = -500;
const int FLOOR_REWARD_VALUE = -10;
const int GOAL_REWARD_VALUE = 500;

void SetupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);

    rewards = new int[mazeRows, mazeColumns];

    for (int i = 0; i < mazeRows; i++)
    {
        for (int j = 0; j < mazeColumns; j++)
        {
            switch (maze[i, j])
            {
                case 0:
                    rewards[i, j] = wallValue;
                    break;
                case 1:
                    rewards[i, j] = floorValue;
                    break;
                case 2:
                    rewards[i, j] = goalValue;
                    break;
            }
        }
    }
}

void SetupQvalue(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);
    qValues = torch.zeros(mazeRows, mazeColumns, 4);
}

bool HasHitWallOrFinishedMaze(int currentRow, int currentColumn, int floorValue)
{
    return rewards[currentRow, currentColumn] != floorValue;
}

long DetermineNextAction(int currentRow, int currentColumn, float epsilon)
{
    Random random = new Random();
    double randomBetweenZeroAndOne = random.NextDouble();
    return randomBetweenZeroAndOne < epsilon ? torch.argmax(qValues[currentRow, currentColumn]).item<long>() : random.Next(4);
}

(int, int) MoveOneSpace(int[,] Maze, int currentRow, int currentColumn, long currentAction)
{
    int nextRow = currentRow;
    int nextColumn = currentColumn;

    if (actions[currentAction] == "up" && currentRow > 0)
    {
        nextRow--;
    }
    else if (actions[currentAction] == "down" && currentRow < Maze.GetLength(0) - 1)
    {
        nextRow++;
    }
    else if (actions[currentAction] == "left" && currentColumn > 0)
    {
        nextColumn--;
    }
    else if (actions[currentAction] == "right" && currentColumn < Maze.GetLength(1) - 1)
    {
        nextColumn++;
    }

    return (nextRow, nextColumn);
}

void TrainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, int episodes)
{
    for (int episode = 0; episode < episodes; episode++)
    {
        Console.WriteLine($"---- Starting episode {episode} ----");
        int currentRow = 11;
        int currentColumn = 5;

        while (!HasHitWallOrFinishedMaze(currentRow, currentColumn, floorValue))
        {
            long currentAction = DetermineNextAction(currentRow, currentColumn, epsilon);
            int prevRow = currentRow;
            int prevColumn = currentColumn;

            (int, int) nextMove = MoveOneSpace(maze, currentRow, currentColumn, currentAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;

            float reward = rewards[currentRow, currentColumn];
            float prevQValue = qValues[prevRow, prevColumn, currentAction].item<float>();
            float temporalDifference = reward + (discountFactor * torch.max(qValues[currentRow, currentColumn])).item<float>() - prevQValue;
            qValues[prevRow, prevColumn, currentAction] = prevQValue + (learningRate * temporalDifference);
        }

        Console.WriteLine($"---- Finished episode {episode} ----");
    }

    Console.WriteLine("Completed training");
}
List<int[]> navigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
{
    List<int[]> path = new List<int[]>();
    if (HasHitWallOrFinishedMaze(startRow, startColumn, floorValue))
    {
        return new List<int[]>();
    }
    else
    {
        int currentRow = startRow;
        int currentColumn = startColumn;
        path = new List<int[]> { new int[] { currentRow, currentColumn } };

        while (!HasHitWallOrFinishedMaze(currentRow, currentColumn, floorValue))
        {
            int nextAction = (int)DetermineNextAction(currentRow, currentColumn, 1.0f);
            (int, int) nextMove = MoveOneSpace(maze, currentRow, currentColumn, nextAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;

            if (rewards[currentRow, currentColumn] != wallValue)
            {
                path.Add(new int[] { currentRow, currentColumn });
            }
            else
            {
                continue;
            }
        }
    }

    int moveCount = 1;
    for (int i = 0; i < path.Count; i++)
    {
        Console.Write("Move " + moveCount + ":(");
        foreach (int element in path[i])
        {
            Console.Write("" + element);
        }
        Console.Write(" )");
        Console.WriteLine();
        moveCount++;
    }
    return path;
}



const float EPSILON = 0.95f;
const float DISCOUNT_FACTOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const int EPISODES = 1500;
const int START_ROW = 11;
const int START_COLUMN = 5;

SetupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);
SetupQvalue(maze1);
TrainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES);
navigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);



