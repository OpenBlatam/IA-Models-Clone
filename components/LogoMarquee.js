import Marquee from "react-fast-marquee";

const logos = [
  "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
  "https://upload.wikimedia.org/wikipedia/commons/a/ab/Apple-logo.png",
  "https://upload.wikimedia.org/wikipedia/commons/5/51/Google.png",
  "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
  "https://upload.wikimedia.org/wikipedia/commons/6/6e/Amazon_logo.svg",
  "https://upload.wikimedia.org/wikipedia/commons/2/2f/Meta_Platforms_Logo.svg",
  "https://upload.wikimedia.org/wikipedia/commons/6/6b/Spotify_logo_with_text.svg",
  "https://upload.wikimedia.org/wikipedia/commons/2/2f/Instagram_logo.svg",
  "https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg",
  "https://upload.wikimedia.org/wikipedia/commons/9/96/Twitter_bird_logo_2012.svg"
];

export default function LogoMarquee() {
  return (
    <div className="py-8 bg-white">
      <h2 className="text-center text-xl font-medium mb-6">
        World-class marketing teams trust Jasper
      </h2>
      <Marquee gradient={false} speed={40} pauseOnHover={true}>
        {logos.map((src, i) => (
          <img
            key={i}
            src={src}
            alt="Logo"
            className="mx-12 h-12 object-contain grayscale opacity-80 hover:opacity-100 transition"
            style={{ maxHeight: 60 }}
          />
        ))}
      </Marquee>
    </div>
  );
} 