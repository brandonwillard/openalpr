#ifndef OPENALPR_VIDEOBUFFER_H
#define OPENALPR_VIDEOBUFFER_H

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <queue>

#include "opencv2/highgui/highgui.hpp"

#include "support/filesystem.h"
#include "support/tinythread.h"
#include "support/platform.h"

#include <boost/atomic.hpp>


template<typename T, size_t Size>
class ringbuffer {
  public:
    ringbuffer() : head_(0), tail_(0) {}

    bool push(const T & value)
    {
      size_t head = head_.load(boost::memory_order_relaxed);
      size_t next_head = next(head);
      if (next_head == tail_.load(boost::memory_order_acquire))
        return false;
      ring_[head] = value;
      head_.store(next_head, boost::memory_order_release);
      return true;
    }

    bool pop(T & value)
    {
      size_t tail = tail_.load(boost::memory_order_relaxed);
      if (tail == head_.load(boost::memory_order_acquire))
        return false;
      value = ring_[tail];
      tail_.store(next(tail), boost::memory_order_release);
      return true;
    }

  private:
    size_t next(size_t current)
    {
      return (current + 1) % Size;
    }

    T ring_[Size];
    boost::atomic<size_t> head_, tail_;
};

/*
 * TODO: Consider adding a thread pool: https://stackoverflow.com/a/19500405/3006474.
 */
class VideoDispatcher
{
  public:

    bool active;
    
    int latestFrameNumber;
    int lastFrameRead;
    int fps;
    
    std::string mjpeg_url;

    /*! 
     * \param mjpeg_url URL for MJPEG feed.
     * \param fps Frames-per-second.  Does nothing. 
     */
    VideoDispatcher(std::string mjpeg_url, int fps)
    {
      this->active = true;
      this->latestFrameNumber = -1;
      this->lastFrameRead = -1;
      this->fps = fps;
      this->mjpeg_url = mjpeg_url;
    }
    
    int getLatestFrame(cv::Mat* frame)
    {
      /* tthread::lock_guard<tthread::mutex> guard(mMutex); */
      
      // XXX: We lose some safety with these counts?  (Doesn't really matter,
      // though.)
      if (latestFrameNumber == lastFrameRead)
        return -1;

      bool popped = this->frameBuffer.pop(*frame);

      if (!popped)
        return -1;

      this->lastFrameRead = this->latestFrameNumber;
      
      return this->lastFrameRead;
    }
    
    void setLatestFrame(cv::Mat& frame)
    {      
      /* tthread::lock_guard<tthread::mutex> guard(mMutex); */

      this->frameBuffer.push(frame.clone());

      // std::stringstream ss;
      // ss << "Frame buffer size: " << this->frameBuffer.size();
      // this->log_info(ss.str());

      this->latestFrameNumber++;
    }
    
    virtual void log_info(std::string message)
    {
      std::cout << message << std::endl;
    }

    virtual void log_error(std::string error)
    {
      std::cerr << error << std::endl;
    }
    
  private:

    /* tthread::mutex mMutex; */
    
    ringbuffer<cv::Mat, 60> frameBuffer;
};

class VideoBuffer
{

  public:
    VideoBuffer();
    virtual ~VideoBuffer();

    void connect(std::string mjpeg_url, int fps);

    /*
     If a new frame is available, the function sets "frame" to it and returns
     the frame number.  If no frames are available, or the latest has already
     been grabbed, it returns -1.  `regionsOfInterest` is set to a list of good
     regions to check for license plates.  Default is one rectangle for the
     entire frame.
    */
    int getLatestFrame(cv::Mat* frame);

    void disconnect();
    
  protected:
  
    virtual VideoDispatcher* createDispatcher(std::string mjpeg_url, int fps);
    
  private:
    
    VideoDispatcher* dispatcher;
};




#endif // OPENALPR_VIDEOBUFFER_H
