import { motion } from "framer-motion";
import { Icons } from "@/components/shared/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface GameStats {
  currentStreak: number;
  totalXP: number;
  level: number;
  lessonsCompleted: number;
  dailyGoal: number;
  dailyProgress: number;
}

export function GameStats({ stats }: { stats: GameStats }) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Current Streak</CardTitle>
          <Icons.flame className="h-4 w-4 text-orange-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.currentStreak} days</div>
          <p className="text-xs text-muted-foreground">
            Keep it up! 🔥
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total XP</CardTitle>
          <Icons.star className="h-4 w-4 text-yellow-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.totalXP} XP</div>
          <p className="text-xs text-muted-foreground">
            Level {stats.level}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Lessons Completed</CardTitle>
          <Icons.trophy className="h-4 w-4 text-blue-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.lessonsCompleted}</div>
          <p className="text-xs text-muted-foreground">
            Great progress!
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Daily Goal</CardTitle>
          <Icons.target className="h-4 w-4 text-green-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.dailyProgress}/{stats.dailyGoal} XP</div>
          <Progress value={(stats.dailyProgress / stats.dailyGoal) * 100} className="mt-2" />
        </CardContent>
      </Card>
    </div>
  );
}

export function GameProgress() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Weekly Progress</CardTitle>
        </CardHeader>
        <CardContent className="pl-2">
          <div className="h-[200px] w-full">
            {/* Add a chart component here */}
            <div className="flex h-full items-center justify-center text-muted-foreground">
              Weekly progress chart will be displayed here
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="col-span-3">
        <CardHeader>
          <CardTitle>Recent Achievements</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center">
              <Icons.trophy className="mr-2 h-4 w-4 text-yellow-500" />
              <div>
                <p className="text-sm font-medium">Perfect Week</p>
                <p className="text-xs text-muted-foreground">Complete 7 days in a row</p>
              </div>
            </div>
            <div className="flex items-center">
              <Icons.star className="mr-2 h-4 w-4 text-blue-500" />
              <div>
                <p className="text-sm font-medium">Quick Learner</p>
                <p className="text-xs text-muted-foreground">Complete 5 lessons in one day</p>
              </div>
            </div>
            <div className="flex items-center">
              <Icons.flame className="mr-2 h-4 w-4 text-orange-500" />
              <div>
                <p className="text-sm font-medium">On Fire</p>
                <p className="text-xs text-muted-foreground">Maintain a 3-day streak</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}    