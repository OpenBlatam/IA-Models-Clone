import { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import DateTimePicker from '@react-native-community/datetimepicker';
import { useCreatePost } from '@/hooks/useApi';
import { postSchema, type PostFormData } from '@/lib/validation';
import { Input } from '@/components/ui/Input';
import { TextArea } from '@/components/ui/TextArea';
import { Checkbox } from '@/components/ui/Checkbox';
import { Button } from '@/components/ui/Button';
import { PLATFORMS } from '@/utils/constants';
import { Ionicons } from '@expo/vector-icons';
import { format } from 'date-fns';
import { showToast } from '@/utils/toast';

export default function CreatePostScreen() {
  const router = useRouter();
  const createPost = useCreatePost();
  const [showDatePicker, setShowDatePicker] = useState(false);

  const {
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
    setValue,
  } = useForm<PostFormData>({
    resolver: zodResolver(postSchema),
    defaultValues: {
      content: '',
      platforms: [],
      tags: [],
    },
  });

  const selectedPlatforms = watch('platforms') || [];
  const scheduledTime = watch('scheduled_time');

  const togglePlatform = (platformId: string) => {
    const current = selectedPlatforms;
    if (current.includes(platformId)) {
      setValue('platforms', current.filter((p) => p !== platformId));
    } else {
      setValue('platforms', [...current, platformId]);
    }
  };

  const handleDateChange = (event: any, date?: Date) => {
    setShowDatePicker(false);
    if (date) {
      setValue('scheduled_time', date.toISOString());
    }
  };

  const onSubmit = async (data: PostFormData) => {
    try {
      await createPost.mutateAsync(data);
      showToast.success('Post created successfully');
      router.back();
    } catch (error: any) {
      showToast.error(error.message || 'Failed to create post');
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={24} color="#1f2937" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Create Post</Text>
          <View style={{ width: 24 }} />
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          <View style={styles.content}>
            <Controller
              control={control}
              name="content"
              render={({ field: { onChange, onBlur, value } }) => (
                <TextArea
                  label="Content *"
                  placeholder="Write your post content..."
                  value={value}
                  onChangeText={onChange}
                  onBlur={onBlur}
                  error={errors.content?.message}
                  rows={6}
                />
              )}
            />

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Platforms *</Text>
              <View style={styles.platformsGrid}>
                {PLATFORMS.map((platform) => (
                  <Checkbox
                    key={platform.id}
                    label={platform.name}
                    checked={selectedPlatforms.includes(platform.id)}
                    onPress={() => togglePlatform(platform.id)}
                  />
                ))}
              </View>
              {errors.platforms && (
                <Text style={styles.errorText}>{errors.platforms.message}</Text>
              )}
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Schedule (Optional)</Text>
              <TouchableOpacity
                style={styles.dateButton}
                onPress={() => setShowDatePicker(true)}
              >
                <Ionicons name="calendar-outline" size={20} color="#0ea5e9" />
                <Text style={styles.dateButtonText}>
                  {scheduledTime
                    ? format(new Date(scheduledTime), 'MMM dd, yyyy HH:mm')
                    : 'Select date and time'}
                </Text>
              </TouchableOpacity>
              {showDatePicker && (
                <DateTimePicker
                  value={scheduledTime ? new Date(scheduledTime) : new Date()}
                  mode="datetime"
                  display="default"
                  onChange={handleDateChange}
                  minimumDate={new Date()}
                />
              )}
            </View>

            <Controller
              control={control}
              name="tags"
              render={({ field: { onChange, value } }) => (
                <Input
                  label="Tags (comma separated)"
                  placeholder="tag1, tag2, tag3"
                  value={value?.join(', ') || ''}
                  onChangeText={(text) => {
                    const tags = text
                      .split(',')
                      .map((tag) => tag.trim())
                      .filter((tag) => tag.length > 0);
                    onChange(tags);
                  }}
                  error={errors.tags?.message}
                  helperText="Separate tags with commas"
                />
              )}
            />
          </View>
        </ScrollView>

        <View style={styles.footer}>
          <Button
            title="Create Post"
            onPress={handleSubmit(onSubmit)}
            loading={isSubmitting}
            disabled={isSubmitting}
            style={styles.submitButton}
          />
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  keyboardView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  platformsGrid: {
    gap: 8,
  },
  dateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d1d5db',
  },
  dateButtonText: {
    fontSize: 16,
    color: '#1f2937',
  },
  errorText: {
    fontSize: 12,
    color: '#ef4444',
    marginTop: 4,
  },
  footer: {
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  submitButton: {
    width: '100%',
  },
});

