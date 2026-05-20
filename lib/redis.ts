let redis: any = null;

// Helper function to safely get data from Redis
export async function getFromRedis(key: string): Promise<any> {
  if (process.env.NODE_ENV === 'production') {
    console.log("[REDIS_DISABLED] Redis operations disabled in production");
    return null;
  }
  
  try {
    if (!redis) return null;
    const data = await redis.get(key);
    return data ? JSON.parse(data) : null;
  } catch (error) {
    console.error("[REDIS_GET_ERROR]", error);
    return null;
  }
}

// Helper function to safely set data in Redis
export async function setInRedis(key: string, value: any, expireSeconds?: number): Promise<void> {
  if (process.env.NODE_ENV === 'production') {
    console.log("[REDIS_DISABLED] Redis operations disabled in production");
    return;
  }
  
  try {
    if (!redis) return;
    const stringValue = JSON.stringify(value);
    if (expireSeconds) {
      await redis.set(key, stringValue, "EX", expireSeconds);
    } else {
      await redis.set(key, stringValue);
    }
  } catch (error) {
    console.error("[REDIS_SET_ERROR]", error);
  }
}

// Helper function to safely delete data from Redis
export async function deleteFromRedis(key: string): Promise<void> {
  if (process.env.NODE_ENV === 'production') {
    console.log("[REDIS_DISABLED] Redis operations disabled in production");
    return;
  }
  
  try {
    if (!redis) return;
    await redis.del(key);
  } catch (error) {
    console.error("[REDIS_DELETE_ERROR]", error);
  }
}

export { redis };        