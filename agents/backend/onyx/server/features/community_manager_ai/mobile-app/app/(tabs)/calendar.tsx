import { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Calendar } from 'react-native-calendars';
import { useDailyEvents, useWeeklyEvents } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { format } from 'date-fns';
import type { CalendarEvent } from '@/types';

export default function CalendarScreen() {
  const [selectedDate, setSelectedDate] = useState(format(new Date(), 'yyyy-MM-dd'));
  const { data: dailyEvents, isLoading, refetch } = useDailyEvents(selectedDate);
  const { data: weeklyEvents } = useWeeklyEvents();

  const markedDates: Record<string, any> = {};
  if (weeklyEvents) {
    Object.keys(weeklyEvents).forEach((date) => {
      markedDates[date] = {
        marked: true,
        dotColor: '#0ea5e9',
      };
    });
  }
  markedDates[selectedDate] = {
    ...markedDates[selectedDate],
    selected: true,
    selectedColor: '#0ea5e9',
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.content}>
        <Text style={styles.title}>Calendar</Text>

        <Calendar
          current={selectedDate}
          onDayPress={(day) => setSelectedDate(day.dateString)}
          markedDates={markedDates}
          theme={{
            todayTextColor: '#0ea5e9',
            arrowColor: '#0ea5e9',
            selectedDayBackgroundColor: '#0ea5e9',
            selectedDayTextColor: '#fff',
          }}
        />

        <View style={styles.eventsSection}>
          <Text style={styles.sectionTitle}>
            Events for {format(new Date(selectedDate), 'MMMM dd, yyyy')}
          </Text>

          <ScrollView
            style={styles.eventsList}
            refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
          >
            {isLoading ? (
              <ActivityIndicator size="large" color="#0ea5e9" />
            ) : dailyEvents && dailyEvents.length > 0 ? (
              dailyEvents.map((event) => <EventCard key={event.id} event={event} />)
            ) : (
              <View style={styles.empty}>
                <Text style={styles.emptyText}>No events scheduled for this day</Text>
              </View>
            )}
          </ScrollView>
        </View>
      </View>
    </SafeAreaView>
  );
}

function EventCard({ event }: { event: CalendarEvent }) {
  return (
    <View style={styles.eventCard}>
      <View style={styles.eventHeader}>
        <Text style={styles.eventTime}>
          {format(new Date(event.scheduled_time), 'HH:mm')}
        </Text>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(event.status) }]}>
          <Text style={styles.statusText}>{event.status.toUpperCase()}</Text>
        </View>
      </View>
      <Text style={styles.eventContent} numberOfLines={2}>
        {event.content}
      </Text>
      <View style={styles.eventPlatforms}>
        {event.platforms.map((platform) => (
          <View key={platform} style={styles.platformTag}>
            <Text style={styles.platformText}>{platform}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'published':
      return '#10b981';
    case 'scheduled':
      return '#f59e0b';
    case 'cancelled':
      return '#ef4444';
    default:
      return '#6b7280';
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    flex: 1,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    padding: 16,
    backgroundColor: '#fff',
  },
  eventsSection: {
    flex: 1,
    backgroundColor: '#fff',
    marginTop: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  eventsList: {
    flex: 1,
    padding: 16,
  },
  empty: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 16,
    color: '#9ca3af',
  },
  eventCard: {
    backgroundColor: '#f9fafb',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#0ea5e9',
  },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  eventTime: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
  },
  eventContent: {
    fontSize: 14,
    color: '#4b5563',
    marginBottom: 8,
    lineHeight: 20,
  },
  eventPlatforms: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  platformTag: {
    backgroundColor: '#e0f2fe',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  platformText: {
    fontSize: 11,
    color: '#0369a1',
    fontWeight: '500',
  },
});


