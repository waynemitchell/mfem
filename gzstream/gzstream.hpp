// ============================================================================
// gzstream, C++ iostream classes wrapping the zlib compression library.
// Copyright (C) 2001  Deepak Bandyopadhyay, Lutz Kettner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// ============================================================================
//
// File          : gzstream.h
// Revision      : $Revision: 1.5 $
// Revision_date : $Date: 2002/04/26 23:30:15 $
// Author(s)     : Deepak Bandyopadhyay, Lutz Kettner
// 
// Standard streambuf implementation following Nicolai Josuttis, "The 
// Standard C++ Library".
// ============================================================================

#ifndef GZSTREAM_H
#define GZSTREAM_H 1

// standard C++ with new header file names and std:: namespace
#include <iostream>
#include <fstream>
#include <zlib.h>

namespace mfem {

// ----------------------------------------------------------------------------
// Internal classes to implement gzstream. See below for user classes.
// ----------------------------------------------------------------------------

class gzstreambuf : public std::streambuf {
private:
    static const int bufferSize = 8192;    // size of data buff
    // totals 512 bytes under g++ for igzstream at the end.

    gzFile           file;               // file handle for compressed file
    char             buffer[bufferSize]; // data buffer
    char             opened;             // open/close state of stream
    char             mode[16];           // I/O mode chars

    int flush_buffer();
public:
    gzstreambuf() : opened(0) {
        setp( buffer, buffer + (bufferSize-1));
        setg( buffer + 4,     // beginning of putback area
              buffer + 4,     // read position
              buffer + 4);    // end position      
        // ASSERT: both input & output capabilities will not be used together
    }
    int is_open() { return opened; }
    gzstreambuf* open(char const *name, char const *mode);
    gzstreambuf* close();
    ~gzstreambuf() { close(); }
    
    virtual int     overflow( int c = EOF);
    virtual int     underflow();
    virtual int     sync();
};

class gzstreambase : virtual public std::ios {
protected:
    gzstreambuf buf;
public:
    gzstreambase() { init(&buf); }
    gzstreambase(char const * name, char const *mode);
    ~gzstreambase();
    void open(char const * name, char const *mode);
    void close();
    gzstreambuf* rdbuf() { return &buf; }
};

// ----------------------------------------------------------------------------
// User classes. Use igzstream and ogzstream analogously to ifstream and
// ofstream respectively. They read and write files based on the gz* 
// function interface of the zlib. Files are compatible with gzip compression.
// ----------------------------------------------------------------------------

class igzstream : public gzstreambase, public std::istream {
public:
    igzstream() : std::istream( &buf) {} 
   /** Constructs a gzip compressed input stream object and opens
       the associated file. The mode chars are as in gzopen(),
       https://refspecs.linuxbase.org/LSB_3.0.0/LSB-PDA/LSB-PDA/zlib-gzopen-1.html.
       Default mode is "rb". */
    igzstream(char const * name, char const *mode = "rb")
        : gzstreambase(name, mode), std::istream( &buf) {}  
    gzstreambuf* rdbuf() { return gzstreambase::rdbuf(); }
   /** Opens a gzip compressed input stream. The mode chars are as in gzopen(),
       https://refspecs.linuxbase.org/LSB_3.0.0/LSB-PDA/LSB-PDA/zlib-gzopen-1.html.
       Default mode is "rb". */
    void open(char const * name, char const *mode = "rb") {
        gzstreambase::open(name, mode);
    }
};

class ogzstream : public gzstreambase, public std::ostream {
public:
    ogzstream() : std::ostream( &buf) {}
   /** Constructs a gzip compressed output stream object and opens
       the associated file. The mode chars are as in gzopen(),
       https://refspecs.linuxbase.org/LSB_3.0.0/LSB-PDA/LSB-PDA/zlib-gzopen-1.html.
       Default mode is "wb6". */
    ogzstream(char const * name, char const *mode = "wb6")
        : gzstreambase(name, mode), std::ostream( &buf) {}  
    gzstreambuf* rdbuf() { return gzstreambase::rdbuf(); }
   /** Opens a gzip compressed output stream. The mode chars are as in gzopen(),
       https://refspecs.linuxbase.org/LSB_3.0.0/LSB-PDA/LSB-PDA/zlib-gzopen-1.html.
       Default mode is "wb6". */
    void open(char const * name, char const *mode = "wb6") {
        gzstreambase::open(name, mode);
    }
};

} // namespace mfem

#endif // GZSTREAM_H
// ============================================================================
// EOF //

