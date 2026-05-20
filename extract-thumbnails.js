const AWS = require('aws-sdk');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const s3 = new AWS.S3({
  region: 'us-east-1',
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

const BUCKET = 'blatamcursos';
const PREFIX = 'Cursos/';
const THUMBNAIL_PREFIX = 'thumbnails/';

async function listVideos() {
  const res = await s3.listObjectsV2({ Bucket: BUCKET, Prefix: PREFIX }).promise();
  return res.Contents.filter(obj => obj.Key.match(/\.(mp4|mov|mkv|avi)$/i));
}

async function downloadAndExtractThumbnail(key) {
  const fileName = path.basename(key);
  const localVideo = `/tmp/${fileName}`;
  const thumbnailName = fileName.replace(path.extname(fileName), '.jpg');
  const localThumbnail = `/tmp/${thumbnailName}`;

  // Descargar video
  const videoData = await s3.getObject({ Bucket: BUCKET, Key: key }).promise();
  fs.writeFileSync(localVideo, videoData.Body);

  // Extraer thumbnail con ffmpeg (en el segundo 5)
  execSync(`ffmpeg -y -ss 00:00:05 -i "${localVideo}" -frames:v 1 -q:v 2 "${localThumbnail}"`);

  // Subir thumbnail a S3
  const thumbData = fs.readFileSync(localThumbnail);
  await s3.putObject({
    Bucket: BUCKET,
    Key: `${THUMBNAIL_PREFIX}${thumbnailName}`,
    Body: thumbData,
    ContentType: 'image/jpeg',
  }).promise();

  // Limpiar archivos temporales
  fs.unlinkSync(localVideo);
  fs.unlinkSync(localThumbnail);


}

(async () => {
  const videos = await listVideos();
  for (const video of videos) {
    await downloadAndExtractThumbnail(video.Key);
  }
})();  