"""
Test Scheduler
Schedule test runs at specific times or intervals
"""

import schedule
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime
import threading

class TestScheduler:
    """Schedule test runs"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.running = False
        self.scheduled_jobs = []
    
    def schedule_daily(self, time_str: str = "02:00", category: Optional[str] = None):
        """Schedule daily test run"""
        def job():
            print(f"🕐 Scheduled test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._run_tests(category)
        
        schedule.every().day.at(time_str).do(job)
        self.scheduled_jobs.append(('daily', time_str, category))
        print(f"✅ Scheduled daily test run at {time_str}")
    
    def schedule_hourly(self, category: Optional[str] = None):
        """Schedule hourly test run"""
        def job():
            print(f"🕐 Scheduled test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._run_tests(category)
        
        schedule.every().hour.do(job)
        self.scheduled_jobs.append(('hourly', None, category))
        print("✅ Scheduled hourly test run")
    
    def schedule_weekly(self, day: str = "monday", time_str: str = "02:00", category: Optional[str] = None):
        """Schedule weekly test run"""
        def job():
            print(f"🕐 Scheduled test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._run_tests(category)
        
        getattr(schedule.every(), day.lower()).at(time_str).do(job)
        self.scheduled_jobs.append(('weekly', f"{day} {time_str}", category))
        print(f"✅ Scheduled weekly test run on {day} at {time_str}")
    
    def schedule_interval(self, minutes: int, category: Optional[str] = None):
        """Schedule test run at interval"""
        def job():
            print(f"🕐 Scheduled test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._run_tests(category)
        
        schedule.every(minutes).minutes.do(job)
        self.scheduled_jobs.append(('interval', f"{minutes} minutes", category))
        print(f"✅ Scheduled test run every {minutes} minutes")
    
    def _run_tests(self, category: Optional[str] = None):
        """Run tests"""
        cmd = [sys.executable, str(self.project_root / "run_unified_tests.py")]
        if category:
            cmd.append(category)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            print(f"✅ Test run completed with exit code: {result.returncode}")
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def list_scheduled_jobs(self) -> list:
        """List all scheduled jobs"""
        return self.scheduled_jobs
    
    def clear_all_jobs(self):
        """Clear all scheduled jobs"""
        schedule.clear()
        self.scheduled_jobs = []
        print("✅ Cleared all scheduled jobs")
    
    def run_pending(self):
        """Run pending scheduled jobs"""
        schedule.run_pending()
    
    def start(self, blocking: bool = True):
        """Start scheduler"""
        self.running = True
        print("🚀 Test Scheduler Started")
        print(f"📋 Scheduled Jobs: {len(self.scheduled_jobs)}")
        print("Press Ctrl+C to stop")
        print()
        
        if blocking:
            try:
                while self.running:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n⏹️  Scheduler stopped")
                self.running = False
        else:
            # Run in background thread
            def run_scheduler():
                while self.running:
                    schedule.run_pending()
                    time.sleep(60)
            
            thread = threading.Thread(target=run_scheduler, daemon=True)
            thread.start()

def main():
    """Example usage"""
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Scheduler')
    parser.add_argument('--daily', type=str, help='Schedule daily at time (HH:MM)')
    parser.add_argument('--hourly', action='store_true', help='Schedule hourly')
    parser.add_argument('--weekly', type=str, help='Schedule weekly on day')
    parser.add_argument('--interval', type=int, help='Schedule at interval (minutes)')
    parser.add_argument('--category', type=str, help='Test category to run')
    parser.add_argument('--list', action='store_true', help='List scheduled jobs')
    parser.add_argument('--clear', action='store_true', help='Clear all jobs')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    scheduler = TestScheduler(project_root)
    
    if args.clear:
        scheduler.clear_all_jobs()
        return
    
    if args.list:
        jobs = scheduler.list_scheduled_jobs()
        print(f"Scheduled Jobs: {len(jobs)}")
        for job in jobs:
            print(f"  • {job[0]}: {job[1]}")
        return
    
    if args.daily:
        scheduler.schedule_daily(args.daily, args.category)
    elif args.hourly:
        scheduler.schedule_hourly(args.category)
    elif args.weekly:
        scheduler.schedule_weekly(args.weekly, category=args.category)
    elif args.interval:
        scheduler.schedule_interval(args.interval, args.category)
    else:
        # Default: schedule daily at 2 AM
        scheduler.schedule_daily("02:00", args.category)
    
    # Start scheduler
    scheduler.start()

if __name__ == "__main__":
    main()







