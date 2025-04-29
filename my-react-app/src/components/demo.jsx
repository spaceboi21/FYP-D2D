// import React, { useRef } from 'react';

// const Demo = () => {
//   const videoRef = useRef(null);

//   // Function to scroll to and play the video
//   const handleWatchNow = () => {
//     if (videoRef.current) {
//       // Scroll to video smoothly
//       videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
//       // Play video when button is clicked
//       videoRef.current.play().catch((err) => {
//         console.error('Error playing the video:', err); // In case autoplay fails
//       });
//     }
//   };

//   return (
//     <div className='w-full py-16 px-4' style={{ backgroundColor: '#E7E4EC' }}>
//       <div className='max-w-[1240px] mx-auto grid md:grid-cols-2'>
//         <div className='flex flex-col justify-center'>
//           <p className='text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 font-bold'>
//             WATCH OUR INTRO VIDEO
//           </p>
//           <h1 className='md:text-4xl sm:text-3xl text-2xl font-bold py-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'>
//             Learn More About Our Product
//           </h1>
//           <p className='text-gray-700 font-medium'>
//             Watch our short intro video to understand how our platform can help you drive better results with real-time analytics.
//           </p>
//           <button
//             className='w-[200px] rounded-md font-medium my-6 mx-auto md:mx-0 py-3 text-white transition bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 shadow-md'
//             onClick={handleWatchNow}
//           >
//             Watch Now
//           </button>
//         </div>

//         <div className='flex justify-center'>
//           <iframe
//             width="80%"
//             height="400"
//             style={styles.video}
//             src="https://www.youtube.com/embed/ny2rBxwvxBY"
//             title="YouTube video player"
//             frameBorder="0"
//             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
//             allowFullScreen
//           ></iframe>
//         </div>
//       </div>
//     </div>
//   );
// };

// const styles = {
//   video: {
//     width: '80%',
//     maxWidth: '700px',
//     borderRadius: '8px',
//     margin: 'auto',
//   },
// };

// export default Demo;
import React, { useRef, useState } from 'react';

const Demo = () => {
  const videoRef = useRef(null);
  const [iframeSrc, setIframeSrc] = useState("https://www.youtube.com/embed/ny2rBxwvxBY");

  // Function to scroll to and play the video
  const handleWatchNow = () => {
    if (videoRef.current) {
      videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    // Add autoplay=1 to the YouTube embed URL
    setIframeSrc("https://www.youtube.com/embed/ny2rBxwvxBY?autoplay=1");
  };

  return (
    <div className='w-full py-16 px-4' style={{ backgroundColor: '#E7E4EC' }}>
      <div className='max-w-[1240px] mx-auto grid md:grid-cols-2'>
        <div className='flex flex-col justify-center'>
          <p className='text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 font-bold'>
            WATCH OUR INTRO VIDEO
          </p>
          <h1 className='md:text-4xl sm:text-3xl text-2xl font-bold py-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'>
            Learn More About Our Product
          </h1>
          <p className='text-gray-700 font-medium'>
            Watch our short intro video to understand how our platform can help you drive better results with real-time analytics.
          </p>
          <button
            className='w-[200px] rounded-md font-medium my-6 mx-auto md:mx-0 py-3 text-white transition bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 shadow-md'
            onClick={handleWatchNow}
          >
            Watch Now
          </button>
        </div>

        <div className='flex justify-center'>
          <iframe
            ref={videoRef}
            width="80%"
            height="400"
            style={styles.video}
            src={iframeSrc}
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        </div>
      </div>
    </div>
  );
};

const styles = {
  video: {
    width: '80%',
    maxWidth: '700px',
    borderRadius: '8px',
    margin: 'auto',
  },
};

export default Demo;